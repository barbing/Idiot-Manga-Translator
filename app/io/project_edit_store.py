# -*- coding: utf-8 -*-
"""Page-local durable journal for GUI-authored project edits.

The journal is deliberately adjacent to, and independent from, the forward
pipeline checkpoint.  It never republishes ``project.json`` and therefore
cannot race the controller's checkpoint descriptor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
from typing import Any, Iterable, Mapping
import uuid

from app.project_edits.contracts import ProjectEdit
from app.project_edits.ledger import ProjectEditLedger


PROJECT_EDIT_STORE_SCHEMA_VERSION = "project_edit_store_v3"
GENESIS_SHA256 = "0" * 64
_ARTIFACT_CATALOGS = frozenset(
    {
        "cleaned_page_bases",
        "rendered_pages",
        "parent_layers",
        "source_revisions",
        "translation_revisions",
    }
)


class StalePageEditHeadError(RuntimeError):
    """The page changed after an editor command captured its base head."""


class StaleProjectEditHeadError(RuntimeError):
    """The project edit chain changed after a command captured its base."""


def project_edit_store_path(project_path: str) -> str:
    absolute = os.path.abspath(project_path)
    parent = os.path.dirname(absolute) or os.getcwd()
    return os.path.join(parent, f".{os.path.basename(absolute)}.gui-edits.sqlite3")


def inspect_project_edit_store(project_path: str) -> dict[str, str] | None:
    """Read store identity without creating or mutating the sidecar."""

    store_path = project_edit_store_path(project_path)
    if not os.path.isfile(store_path):
        return None
    connection = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT key, value FROM edit_store_meta"
        ).fetchall()
    finally:
        connection.close()
    metadata = {str(key): str(value) for key, value in rows}
    if metadata.get("schema_version") != PROJECT_EDIT_STORE_SCHEMA_VERSION:
        raise ValueError("unsupported project edit store schema")
    if not metadata.get("project_id"):
        raise ValueError("project edit store identity is missing")
    return metadata


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, field_name: str) -> str:
    candidate = str(value or "").lower()
    if len(candidate) != 64 or any(
        character not in "0123456789abcdef" for character in candidate
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return candidate


def _normalize_artifact_catalogs(
    catalogs: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[str, dict[str, Any]]]]:
    if not isinstance(catalogs, Mapping):
        raise TypeError("base artifact revisions must be a mapping")
    normalized: dict[str, list[dict[str, Any]]] = {}
    by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for catalog in sorted(_ARTIFACT_CATALOGS):
        values = catalogs.get(catalog)
        if values is None and catalog in {
            "source_revisions",
            "translation_revisions",
        }:
            values = ()
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"base artifact catalog {catalog} must be a list")
        records: list[dict[str, Any]] = []
        for value in values:
            if not isinstance(value, Mapping):
                raise ValueError("base artifact revisions must be mappings")
            record = dict(value)
            revision_id = str(record.get("revision_id") or "").strip()
            page_id = str(record.get("page_id") or "").strip()
            if not revision_id or not page_id:
                raise ValueError("base artifact revision identity is invalid")
            if revision_id in by_id:
                raise ValueError(
                    f"base artifact revision identity is duplicated: {revision_id}"
                )
            records.append(record)
            by_id[revision_id] = (catalog, record)
        normalized[catalog] = sorted(
            records,
            key=lambda record: str(record["revision_id"]),
        )
    return normalized, by_id


def _transaction_sha256(
    *,
    previous_global_sha256: str,
    previous_page_sha256: str,
    transaction_id: str,
    project_id: str,
    page_id: str,
    automatic_page_sha256: str,
    edit_hashes: tuple[str, ...],
    artifact_hashes: tuple[str, ...],
) -> str:
    return _sha256_bytes(
        _compact_json_bytes(
            {
                "schema_version": PROJECT_EDIT_STORE_SCHEMA_VERSION,
                "previous_global_sha256": previous_global_sha256,
                "previous_page_sha256": previous_page_sha256,
                "transaction_id": transaction_id,
                "project_id": project_id,
                "page_id": page_id,
                "automatic_page_sha256": automatic_page_sha256,
                "edit_hashes": list(edit_hashes),
                "artifact_hashes": list(artifact_hashes),
            }
        )
    )


@dataclass(frozen=True)
class ProjectEditCommitReceipt:
    sequence: int
    transaction_id: str
    page_id: str
    previous_page_head_sha256: str
    transaction_sha256: str
    edit_ids: tuple[str, ...]
    artifact_revision_ids: tuple[str, ...]
    payload_bytes: int
    committed_at: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["edit_ids"] = list(self.edit_ids)
        value["artifact_revision_ids"] = list(self.artifact_revision_ids)
        return value


@dataclass(frozen=True)
class ProjectEditReadSnapshot:
    """One validated materialized project view and its exact edit heads."""

    project: dict[str, Any]
    ledger: ProjectEditLedger
    page_head_sha256: str
    global_head_sha256: str


@dataclass(frozen=True)
class ProjectEditMultiPageReadSnapshot:
    """One materialized project, ledger, and every page/global CAS head."""

    project: dict[str, Any]
    ledger: ProjectEditLedger
    page_head_sha256: tuple[tuple[str, str], ...]
    global_head_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.project, dict):
            raise TypeError("project must be a dictionary")
        if not isinstance(self.ledger, ProjectEditLedger):
            raise TypeError("ledger must be a ProjectEditLedger")
        page_heads = tuple(self.page_head_sha256)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not str(item[0] or "").strip()
            for item in page_heads
        ):
            raise ValueError("page_head_sha256 must contain page/head pairs")
        page_ids = tuple(str(item[0]) for item in page_heads)
        if page_ids != tuple(sorted(page_ids)) or len(page_ids) != len(set(page_ids)):
            raise ValueError("page heads must be unique and sorted by page ID")
        canonical = tuple(
            (
                page_id,
                _require_sha256(str(head), f"page head for {page_id}"),
            )
            for page_id, head in page_heads
        )
        object.__setattr__(self, "page_head_sha256", canonical)
        object.__setattr__(
            self,
            "global_head_sha256",
            _require_sha256(self.global_head_sha256, "global_head_sha256"),
        )

    def page_head(self, page_id: str) -> str:
        identity = str(page_id or "").strip()
        for candidate, head in self.page_head_sha256:
            if candidate == identity:
                return head
        raise KeyError(f"project snapshot has no page head: {identity}")


@dataclass(frozen=True)
class ProjectEditPageBatch:
    """One page-local edit transaction inside an all-or-none batch commit."""

    page_id: str
    edits: tuple[ProjectEdit, ...]
    automatic_page_sha256: str
    expected_page_head_sha256: str
    transaction_id: str

    def __post_init__(self) -> None:
        page_id = str(self.page_id or "").strip()
        if not page_id:
            raise ValueError("page_id is required")
        object.__setattr__(self, "page_id", page_id)
        records = tuple(self.edits)
        if not records or any(not isinstance(edit, ProjectEdit) for edit in records):
            raise ValueError("a page edit batch requires ProjectEdit records")
        if any(edit.page_id != page_id for edit in records):
            raise ValueError("page edit batch records must target its page")
        object.__setattr__(self, "edits", records)
        object.__setattr__(
            self,
            "automatic_page_sha256",
            _require_sha256(
                self.automatic_page_sha256,
                "automatic_page_sha256",
            ),
        )
        object.__setattr__(
            self,
            "expected_page_head_sha256",
            _require_sha256(
                self.expected_page_head_sha256,
                "expected_page_head_sha256",
            ),
        )
        transaction_id = str(self.transaction_id or "").strip()
        if not transaction_id or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in transaction_id
        ):
            raise ValueError("edit transaction identity is not path-safe")
        object.__setattr__(self, "transaction_id", transaction_id)


class ProjectEditStore:
    """One project-bound append-only edit journal."""

    def __init__(
        self,
        *,
        project_path: str,
        project_id: str,
        project_origin_sha256: str,
        automated_state_sha256: str,
        base_artifact_revisions: Mapping[str, Any],
        base_ledger: ProjectEditLedger | None = None,
        store_path: str | None = None,
    ) -> None:
        self.project_path = os.path.abspath(project_path)
        expected_store = project_edit_store_path(self.project_path)
        self.store_path = os.path.abspath(store_path or expected_store)
        if self.store_path != os.path.abspath(expected_store):
            raise ValueError(
                "project edit store must use the stable adjacent sidecar path"
            )
        self.project_id = str(project_id or "").strip()
        if not self.project_id:
            raise ValueError("project edit store requires a project ID")
        project_origin_sha256 = _require_sha256(
            project_origin_sha256,
            "project_origin_sha256",
        )
        initial_automated_sha256 = _require_sha256(
            automated_state_sha256,
            "automated_state_sha256",
        )
        requested_base = base_ledger or ProjectEditLedger(
            project_id=self.project_id
        )
        if requested_base.project_id != self.project_id:
            raise ValueError("embedded ledger project identity mismatch")
        requested_artifacts, requested_artifact_index = (
            _normalize_artifact_catalogs(base_artifact_revisions)
        )
        self._connection = sqlite3.connect(
            self.store_path,
            timeout=30.0,
            isolation_level=None,
        )
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._closed = False
        self._initialize(
            project_origin_sha256,
            initial_automated_sha256,
            requested_base,
            requested_artifacts,
        )
        self._base_artifact_index = requested_artifact_index
        if requested_base.edits:
            loaded = self.load_ledger()
            requested_records = tuple(
                record.to_dict() for record in requested_base.edits
            )
            loaded_records = tuple(record.to_dict() for record in loaded.edits)
            if loaded_records[: len(requested_records)] != requested_records:
                raise ValueError(
                    "embedded ledger does not match the edit-store history"
                )
        for artifact in self.load_artifact_revisions():
            revision_id = str(artifact.get("revision_id") or "")
            catalog = str(artifact.get("catalog") or "")
            stored = dict(artifact)
            stored.pop("catalog", None)
            base = self._base_artifact_index.get(revision_id)
            if base is not None and base != (catalog, stored):
                raise ValueError(
                    f"artifact revision identity conflict: {revision_id}"
                )

    def _initialize(
        self,
        project_origin_sha256: str,
        initial_automated_sha256: str,
        requested_base: ProjectEditLedger,
        requested_artifacts: Mapping[str, Any],
    ) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS edit_store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edit_transactions (
                sequence INTEGER PRIMARY KEY,
                transaction_id TEXT NOT NULL UNIQUE,
                project_id TEXT NOT NULL,
                page_id TEXT NOT NULL,
                automatic_page_sha256 TEXT NOT NULL,
                expected_page_head_sha256 TEXT NOT NULL,
                previous_global_sha256 TEXT NOT NULL,
                previous_page_sha256 TEXT NOT NULL,
                transaction_sha256 TEXT NOT NULL UNIQUE,
                edit_count INTEGER NOT NULL,
                artifact_count INTEGER NOT NULL,
                payload_bytes INTEGER NOT NULL,
                committed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edit_records (
                transaction_sequence INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                edit_id TEXT NOT NULL UNIQUE,
                payload BLOB NOT NULL,
                payload_sha256 TEXT NOT NULL,
                PRIMARY KEY (transaction_sequence, ordinal),
                FOREIGN KEY (transaction_sequence)
                    REFERENCES edit_transactions(sequence)
            );
            CREATE TABLE IF NOT EXISTS artifact_revision_records (
                transaction_sequence INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                revision_id TEXT NOT NULL UNIQUE,
                page_id TEXT NOT NULL,
                catalog TEXT NOT NULL,
                payload BLOB NOT NULL,
                payload_sha256 TEXT NOT NULL,
                PRIMARY KEY (transaction_sequence, ordinal),
                FOREIGN KEY (transaction_sequence)
                    REFERENCES edit_transactions(sequence)
            );
            CREATE TABLE IF NOT EXISTS page_edit_heads (
                page_id TEXT PRIMARY KEY,
                transaction_sequence INTEGER NOT NULL,
                transaction_sha256 TEXT NOT NULL,
                FOREIGN KEY (transaction_sequence)
                    REFERENCES edit_transactions(sequence)
            );
            """
        )
        existing = dict(
            self._connection.execute(
                "SELECT key, value FROM edit_store_meta"
            ).fetchall()
        )
        expected = {
            "schema_version": PROJECT_EDIT_STORE_SCHEMA_VERSION,
            "project_id": self.project_id,
            "project_origin_sha256": project_origin_sha256,
        }
        if existing:
            for key, value in expected.items():
                if str(existing.get(key) or "") != value:
                    raise ValueError(f"project edit store binding mismatch: {key}")
            # The initial automated fingerprint is provenance, not a lifetime
            # lock: a live forward checkpoint can append newly committed pages.
            _require_sha256(
                str(existing.get("initial_automated_state_sha256") or ""),
                "stored initial automated state",
            )
            payload = str(existing.get("base_ledger_payload") or "")
            stored_base = ProjectEditLedger.from_dict(json.loads(payload))
            stored_fingerprint = str(existing.get("base_ledger_sha256") or "")
            if stored_base.fingerprint() != stored_fingerprint:
                raise ValueError("stored embedded-ledger fingerprint mismatch")
            artifacts_payload = str(
                existing.get("base_artifact_revisions_payload") or ""
            )
            stored_artifacts = json.loads(artifacts_payload)
            stored_artifact_sha256 = str(
                existing.get("base_artifact_revisions_sha256") or ""
            )
            if _sha256_bytes(_compact_json_bytes(stored_artifacts)) != stored_artifact_sha256:
                raise ValueError("stored base-artifact fingerprint mismatch")
            _, stored_artifact_index = _normalize_artifact_catalogs(
                stored_artifacts
            )
            _, requested_artifact_index = _normalize_artifact_catalogs(
                requested_artifacts
            )
            for revision_id, stored in stored_artifact_index.items():
                if requested_artifact_index.get(revision_id) != stored:
                    raise ValueError(
                        f"embedded artifact revision changed: {revision_id}"
                    )
            self._base_ledger = stored_base
            return
        base_payload = json.dumps(
            requested_base.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        artifact_payload = json.dumps(
            requested_artifacts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for key, value in (
                *expected.items(),
                ("initial_automated_state_sha256", initial_automated_sha256),
                ("base_ledger_sha256", requested_base.fingerprint()),
                ("base_ledger_payload", base_payload),
                (
                    "base_artifact_revisions_sha256",
                    _sha256_bytes(artifact_payload.encode("utf-8")),
                ),
                ("base_artifact_revisions_payload", artifact_payload),
                ("created_at", _utc_now()),
            ):
                self._connection.execute(
                    "INSERT INTO edit_store_meta(key, value) VALUES (?, ?)",
                    (key, value),
                )
            self._connection.execute("COMMIT")
            self._base_ledger = requested_base
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("project edit store is closed")

    def page_head(self, page_id: str) -> str:
        self._require_open()
        page_id = str(page_id or "").strip()
        if not page_id:
            raise ValueError("page_id is required")
        row = self._connection.execute(
            "SELECT transaction_sha256 FROM page_edit_heads WHERE page_id = ?",
            (page_id,),
        ).fetchone()
        return str(row[0]) if row is not None else GENESIS_SHA256

    def global_head(self) -> str:
        self._require_open()
        row = self._connection.execute(
            "SELECT transaction_sha256 FROM edit_transactions "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        return str(row[0]) if row is not None else GENESIS_SHA256

    def _rows_for_transaction(
        self,
        sequence: int,
    ) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        edits = self._connection.execute(
            "SELECT ordinal, edit_id, payload, payload_sha256 "
            "FROM edit_records WHERE transaction_sequence = ? ORDER BY ordinal",
            (sequence,),
        ).fetchall()
        artifacts = self._connection.execute(
            "SELECT ordinal, revision_id, page_id, catalog, payload, payload_sha256 "
            "FROM artifact_revision_records WHERE transaction_sequence = ? "
            "ORDER BY ordinal",
            (sequence,),
        ).fetchall()
        return edits, artifacts

    def _load_validated(
        self,
    ) -> tuple[ProjectEditLedger, tuple[dict[str, Any], ...]]:
        self._require_open()
        transactions = self._connection.execute(
            """
            SELECT sequence, transaction_id, project_id, page_id,
                   automatic_page_sha256, expected_page_head_sha256,
                   previous_global_sha256, previous_page_sha256,
                   transaction_sha256, edit_count, artifact_count
            FROM edit_transactions ORDER BY sequence
            """
        ).fetchall()
        previous_global = GENESIS_SHA256
        page_heads: dict[str, str] = {}
        edits: list[ProjectEdit] = list(self._base_ledger.edits)
        artifacts: list[dict[str, Any]] = []
        for expected_sequence, row in enumerate(transactions):
            (
                sequence,
                transaction_id,
                project_id,
                page_id,
                automatic_page_sha256,
                expected_page_head,
                stored_previous_global,
                stored_previous_page,
                stored_transaction_sha256,
                edit_count,
                artifact_count,
            ) = row
            if int(sequence) != expected_sequence:
                raise ValueError("project edit transaction sequence is not contiguous")
            if str(project_id) != self.project_id:
                raise ValueError("project edit transaction identity mismatch")
            page_id = str(page_id)
            previous_page = page_heads.get(page_id, GENESIS_SHA256)
            if str(stored_previous_global) != previous_global:
                raise ValueError(f"global edit chain mismatch at {sequence}")
            if str(stored_previous_page) != previous_page:
                raise ValueError(f"page edit chain mismatch at {sequence}")
            if str(expected_page_head) != previous_page:
                raise ValueError(f"stored page-head expectation mismatch at {sequence}")
            automatic_page_sha256 = _require_sha256(
                str(automatic_page_sha256),
                "stored automatic page fingerprint",
            )
            edit_rows, artifact_rows = self._rows_for_transaction(int(sequence))
            if len(edit_rows) != int(edit_count) or len(artifact_rows) != int(artifact_count):
                raise ValueError(f"project edit transaction cardinality mismatch at {sequence}")
            edit_hashes: list[str] = []
            artifact_hashes: list[str] = []
            for expected_ordinal, edit_row in enumerate(edit_rows):
                ordinal, edit_id, payload, payload_sha256 = edit_row
                if int(ordinal) != expected_ordinal:
                    raise ValueError(f"edit ordinal mismatch at {sequence}")
                payload_bytes = bytes(payload)
                if _sha256_bytes(payload_bytes) != str(payload_sha256):
                    raise ValueError(f"project edit payload hash mismatch at {sequence}")
                value = json.loads(payload_bytes.decode("utf-8"))
                edit = ProjectEdit.from_persisted_dict(value)
                if (
                    edit.edit_id != str(edit_id)
                    or edit.page_id != page_id
                    or edit.project_id != self.project_id
                ):
                    raise ValueError(f"project edit identity mismatch at {sequence}")
                edits.append(edit)
                edit_hashes.append(str(payload_sha256))
            for expected_ordinal, artifact_row in enumerate(artifact_rows):
                ordinal, revision_id, artifact_page_id, catalog, payload, payload_sha256 = artifact_row
                if int(ordinal) != expected_ordinal:
                    raise ValueError(f"artifact ordinal mismatch at {sequence}")
                payload_bytes = bytes(payload)
                if _sha256_bytes(payload_bytes) != str(payload_sha256):
                    raise ValueError(f"artifact payload hash mismatch at {sequence}")
                value = json.loads(payload_bytes.decode("utf-8"))
                if (
                    str(value.get("revision_id") or "") != str(revision_id)
                    or str(value.get("page_id") or "") != page_id
                    or str(artifact_page_id) != page_id
                    or str(value.get("catalog") or "") != str(catalog)
                ):
                    raise ValueError(f"artifact revision identity mismatch at {sequence}")
                artifacts.append(value)
                artifact_hashes.append(str(payload_sha256))
            expected_transaction_sha256 = _transaction_sha256(
                previous_global_sha256=previous_global,
                previous_page_sha256=previous_page,
                transaction_id=str(transaction_id),
                project_id=self.project_id,
                page_id=page_id,
                automatic_page_sha256=automatic_page_sha256,
                edit_hashes=tuple(edit_hashes),
                artifact_hashes=tuple(artifact_hashes),
            )
            if expected_transaction_sha256 != str(stored_transaction_sha256):
                raise ValueError(f"project edit transaction hash mismatch at {sequence}")
            previous_global = expected_transaction_sha256
            page_heads[page_id] = expected_transaction_sha256
        stored_heads = {
            str(page_id): str(head)
            for page_id, head in self._connection.execute(
                "SELECT page_id, transaction_sha256 FROM page_edit_heads"
            ).fetchall()
        }
        if stored_heads != page_heads:
            raise ValueError("project edit page-head index is inconsistent")
        return ProjectEditLedger(edits, project_id=self.project_id), tuple(artifacts)

    def load_ledger(self) -> ProjectEditLedger:
        return self._load_validated()[0]

    def load_artifact_revisions(self) -> tuple[dict[str, Any], ...]:
        return self._load_validated()[1]

    def commit_page_edits(
        self,
        edits: Iterable[ProjectEdit],
        *,
        automatic_page_sha256: str,
        expected_page_head_sha256: str,
        expected_global_head_sha256: str,
        artifact_revisions: Iterable[Mapping[str, Any]] = (),
        transaction_id: str | None = None,
    ) -> ProjectEditCommitReceipt:
        self._require_open()
        records = tuple(edits)
        artifacts = tuple(dict(value) for value in artifact_revisions)
        if not records and not artifacts:
            raise ValueError("page edit transaction cannot be empty")
        for record in records:
            if not isinstance(record, ProjectEdit):
                raise TypeError("page edit transactions require ProjectEdit records")
            # Compatibility deserialization is read-only evidence.  Revalidate
            # every proposed write through the strict current constructor before
            # deriving payloads or opening a transaction so legacy RGBA fill
            # records cannot be republished as fresh edits.
            ProjectEdit.from_dict(record.to_dict())
        page_ids = {record.page_id for record in records}
        page_ids.update(str(value.get("page_id") or "") for value in artifacts)
        if len(page_ids) != 1 or not next(iter(page_ids), ""):
            raise ValueError("one edit transaction may target only one page")
        page_id = next(iter(page_ids))
        if {record.project_id for record in records} - {self.project_id}:
            raise ValueError("edit transaction project identity mismatch")
        automatic_page_sha256 = _require_sha256(
            automatic_page_sha256,
            "automatic_page_sha256",
        )
        expected_page_head_sha256 = _require_sha256(
            expected_page_head_sha256,
            "expected_page_head_sha256",
        )
        expected_global_head_sha256 = _require_sha256(
            expected_global_head_sha256,
            "expected_global_head_sha256",
        )
        transaction_id = str(transaction_id or uuid.uuid4())
        if not transaction_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in transaction_id
        ):
            raise ValueError("edit transaction identity is not path-safe")

        edit_payloads: list[tuple[str, bytes, str]] = []
        for record in records:
            payload = _compact_json_bytes(record.to_dict())
            edit_payloads.append((record.edit_id, payload, _sha256_bytes(payload)))
        artifact_payloads: list[tuple[str, str, bytes, str]] = []
        for artifact in artifacts:
            revision_id = str(artifact.get("revision_id") or "").strip()
            artifact_page_id = str(artifact.get("page_id") or "").strip()
            catalog = str(artifact.get("catalog") or "").strip()
            if not revision_id or artifact_page_id != page_id:
                raise ValueError("artifact revision identity is invalid")
            if catalog not in _ARTIFACT_CATALOGS:
                raise ValueError("artifact revision catalog is invalid")
            stored_record = dict(artifact)
            stored_record.pop("catalog", None)
            base = self._base_artifact_index.get(revision_id)
            if base is not None and base != (catalog, stored_record):
                raise ValueError(
                    f"artifact revision identity conflict: {revision_id}"
                )
            payload = _compact_json_bytes(artifact)
            artifact_payloads.append(
                (revision_id, catalog, payload, _sha256_bytes(payload))
            )

        committed_at = _utc_now()
        payload_bytes = sum(len(value[1]) for value in edit_payloads) + sum(
            len(value[2]) for value in artifact_payloads
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current_page_head = self.page_head(page_id)
            if current_page_head != expected_page_head_sha256:
                raise StalePageEditHeadError(
                    "page edit head changed before the transaction committed"
                )
            current_ledger = self.load_ledger()
            candidate = current_ledger
            for record in records:
                candidate = candidate.append(record)
            del candidate
            row = self._connection.execute(
                "SELECT sequence, transaction_sha256 FROM edit_transactions "
                "ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            sequence = int(row[0]) + 1 if row is not None else 0
            previous_global = str(row[1]) if row is not None else GENESIS_SHA256
            if previous_global != expected_global_head_sha256:
                raise StaleProjectEditHeadError(
                    "project edit head changed before the transaction committed"
                )
            transaction_sha256 = _transaction_sha256(
                previous_global_sha256=previous_global,
                previous_page_sha256=current_page_head,
                transaction_id=transaction_id,
                project_id=self.project_id,
                page_id=page_id,
                automatic_page_sha256=automatic_page_sha256,
                edit_hashes=tuple(value[2] for value in edit_payloads),
                artifact_hashes=tuple(value[3] for value in artifact_payloads),
            )
            self._connection.execute(
                """
                INSERT INTO edit_transactions(
                    sequence, transaction_id, project_id, page_id,
                    automatic_page_sha256, expected_page_head_sha256,
                    previous_global_sha256, previous_page_sha256,
                    transaction_sha256, edit_count, artifact_count,
                    payload_bytes, committed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sequence,
                    transaction_id,
                    self.project_id,
                    page_id,
                    automatic_page_sha256,
                    expected_page_head_sha256,
                    previous_global,
                    current_page_head,
                    transaction_sha256,
                    len(edit_payloads),
                    len(artifact_payloads),
                    payload_bytes,
                    committed_at,
                ),
            )
            self._connection.executemany(
                "INSERT INTO edit_records(transaction_sequence, ordinal, edit_id, payload, payload_sha256) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (sequence, ordinal, edit_id, payload, payload_sha256)
                    for ordinal, (edit_id, payload, payload_sha256) in enumerate(edit_payloads)
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO artifact_revision_records(
                    transaction_sequence, ordinal, revision_id, page_id,
                    catalog, payload, payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        sequence,
                        ordinal,
                        revision_id,
                        page_id,
                        catalog,
                        payload,
                        payload_sha256,
                    )
                    for ordinal, (revision_id, catalog, payload, payload_sha256)
                    in enumerate(artifact_payloads)
                ],
            )
            self._connection.execute(
                """
                INSERT INTO page_edit_heads(page_id, transaction_sequence, transaction_sha256)
                VALUES (?, ?, ?)
                ON CONFLICT(page_id) DO UPDATE SET
                    transaction_sequence = excluded.transaction_sequence,
                    transaction_sha256 = excluded.transaction_sha256
                """,
                (page_id, sequence, transaction_sha256),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return ProjectEditCommitReceipt(
            sequence=sequence,
            transaction_id=transaction_id,
            page_id=page_id,
            previous_page_head_sha256=current_page_head,
            transaction_sha256=transaction_sha256,
            edit_ids=tuple(value[0] for value in edit_payloads),
            artifact_revision_ids=tuple(value[0] for value in artifact_payloads),
            payload_bytes=payload_bytes,
            committed_at=committed_at,
        )

    def commit_multi_page_edits(
        self,
        batches: Iterable[ProjectEditPageBatch],
        *,
        expected_global_head_sha256: str,
    ) -> tuple[ProjectEditCommitReceipt, ...]:
        """Commit ordered page-local edit transactions all-or-none.

        This is an additive persistence primitive for one explicit GUI command.
        Each stored transaction remains page-local and extends the existing
        global chain; the surrounding SQLite transaction prevents a partial
        project-wide command from becoming durable.
        """

        self._require_open()
        values = tuple(batches)
        if not values or any(not isinstance(item, ProjectEditPageBatch) for item in values):
            raise ValueError("multi-page edit commit requires page batches")
        page_ids = tuple(item.page_id for item in values)
        if page_ids != tuple(sorted(page_ids)) or len(page_ids) != len(set(page_ids)):
            raise ValueError("multi-page edit batches must have unique sorted pages")
        transaction_ids = tuple(item.transaction_id for item in values)
        if len(transaction_ids) != len(set(transaction_ids)):
            raise ValueError("multi-page transaction identities must be unique")
        edit_ids = tuple(edit.edit_id for item in values for edit in item.edits)
        if len(edit_ids) != len(set(edit_ids)):
            raise ValueError("multi-page edit identities must be unique")
        if any(edit.project_id != self.project_id for item in values for edit in item.edits):
            raise ValueError("edit transaction project identity mismatch")
        expected_global_head_sha256 = _require_sha256(
            expected_global_head_sha256,
            "expected_global_head_sha256",
        )
        payloads_by_page: dict[str, tuple[tuple[str, bytes, str], ...]] = {}
        for item in values:
            payloads: list[tuple[str, bytes, str]] = []
            for record in item.edits:
                ProjectEdit.from_dict(record.to_dict())
                payload = _compact_json_bytes(record.to_dict())
                payloads.append((record.edit_id, payload, _sha256_bytes(payload)))
            payloads_by_page[item.page_id] = tuple(payloads)

        committed_at = _utc_now()
        receipts: list[ProjectEditCommitReceipt] = []
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current_global = self.global_head()
            if current_global != expected_global_head_sha256:
                raise StaleProjectEditHeadError(
                    "project edit head changed before the transaction committed"
                )
            current_page_heads = {
                item.page_id: self.page_head(item.page_id) for item in values
            }
            for item in values:
                if current_page_heads[item.page_id] != item.expected_page_head_sha256:
                    raise StalePageEditHeadError(
                        "page edit head changed before the transaction committed"
                    )
            candidate = self.load_ledger()
            for item in values:
                for record in item.edits:
                    candidate = candidate.append(record)
            del candidate

            row = self._connection.execute(
                "SELECT sequence FROM edit_transactions ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            next_sequence = int(row[0]) + 1 if row is not None else 0
            previous_global = current_global
            for batch_index, item in enumerate(values):
                sequence = next_sequence + batch_index
                previous_page = current_page_heads[item.page_id]
                payloads = payloads_by_page[item.page_id]
                payload_bytes = sum(len(value[1]) for value in payloads)
                transaction_sha256 = _transaction_sha256(
                    previous_global_sha256=previous_global,
                    previous_page_sha256=previous_page,
                    transaction_id=item.transaction_id,
                    project_id=self.project_id,
                    page_id=item.page_id,
                    automatic_page_sha256=item.automatic_page_sha256,
                    edit_hashes=tuple(value[2] for value in payloads),
                    artifact_hashes=(),
                )
                self._connection.execute(
                    """
                    INSERT INTO edit_transactions(
                        sequence, transaction_id, project_id, page_id,
                        automatic_page_sha256, expected_page_head_sha256,
                        previous_global_sha256, previous_page_sha256,
                        transaction_sha256, edit_count, artifact_count,
                        payload_bytes, committed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sequence,
                        item.transaction_id,
                        self.project_id,
                        item.page_id,
                        item.automatic_page_sha256,
                        item.expected_page_head_sha256,
                        previous_global,
                        previous_page,
                        transaction_sha256,
                        len(payloads),
                        0,
                        payload_bytes,
                        committed_at,
                    ),
                )
                self._connection.executemany(
                    "INSERT INTO edit_records(transaction_sequence, ordinal, edit_id, payload, payload_sha256) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (sequence, ordinal, edit_id, payload, payload_sha256)
                        for ordinal, (edit_id, payload, payload_sha256) in enumerate(payloads)
                    ],
                )
                self._connection.execute(
                    """
                    INSERT INTO page_edit_heads(
                        page_id, transaction_sequence, transaction_sha256
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(page_id) DO UPDATE SET
                        transaction_sequence = excluded.transaction_sequence,
                        transaction_sha256 = excluded.transaction_sha256
                    """,
                    (item.page_id, sequence, transaction_sha256),
                )
                receipts.append(
                    ProjectEditCommitReceipt(
                        sequence=sequence,
                        transaction_id=item.transaction_id,
                        page_id=item.page_id,
                        previous_page_head_sha256=previous_page,
                        transaction_sha256=transaction_sha256,
                        edit_ids=tuple(value[0] for value in payloads),
                        artifact_revision_ids=(),
                        payload_bytes=payload_bytes,
                        committed_at=committed_at,
                    )
                )
                previous_global = transaction_sha256
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        return tuple(receipts)

    def materialize_project(self, project: Mapping[str, Any]) -> dict[str, Any]:
        self._require_open()
        from app.io.project import migrate_project_schema_v2, validate_project_schema_v2

        migrated = migrate_project_schema_v2(project, project_id=self.project_id)
        materialized = dict(migrated)
        embedded_ledger = ProjectEditLedger.from_dict(migrated["edit_ledger"])
        stored_ledger = self.load_ledger()
        embedded_records = tuple(
            record.to_dict() for record in embedded_ledger.edits
        )
        stored_records = tuple(record.to_dict() for record in stored_ledger.edits)
        if embedded_records and stored_records[: len(embedded_records)] != embedded_records:
            raise ValueError(
                "project edit sidecar does not extend the embedded ledger"
            )
        if len(stored_records) < len(embedded_records):
            raise ValueError(
                "project edit sidecar is missing embedded ledger history"
            )
        materialized["edit_ledger"] = stored_ledger.to_dict()
        catalogs = {
            key: list(value)
            for key, value in dict(materialized["artifact_revisions"]).items()
        }
        catalogs.setdefault("source_revisions", [])
        catalogs.setdefault("translation_revisions", [])
        by_id = {
            str(value.get("revision_id") or ""): (catalog, dict(value))
            for catalog, values in catalogs.items()
            for value in values
            if isinstance(value, Mapping)
        }
        for artifact in self.load_artifact_revisions():
            revision_id = str(artifact.get("revision_id") or "")
            catalog = str(artifact.get("catalog") or "")
            stored = dict(artifact)
            stored.pop("catalog", None)
            if revision_id in by_id:
                existing_catalog, existing = by_id[revision_id]
                if existing_catalog != catalog or existing != stored:
                    raise ValueError(
                        f"artifact revision identity conflict: {revision_id}"
                    )
                continue
            catalogs[catalog].append(stored)
            by_id[revision_id] = (catalog, stored)
        materialized["artifact_revisions"] = catalogs
        validate_project_schema_v2(materialized)
        return materialized

    def materialize_project_snapshot(
        self,
        project: Mapping[str, Any],
        *,
        page_id: str,
    ) -> ProjectEditReadSnapshot:
        """Read materialized state and both heads from one SQLite snapshot.

        The explicit deferred transaction fixes the WAL read view on the first
        store query.  A concurrent writer may finish while this method runs,
        but its ledger, artifacts, and heads cannot be mixed into the returned
        value.  The method never creates or mutates a journal record.
        """

        self._require_open()
        page_id = str(page_id or "").strip()
        if not page_id:
            raise ValueError("page_id is required")
        if self._connection.in_transaction:
            raise RuntimeError("project edit store already has an active transaction")
        self._connection.execute("BEGIN DEFERRED")
        try:
            materialized = self.materialize_project(project)
            ledger = ProjectEditLedger.from_dict(materialized["edit_ledger"])
            page_head_sha256 = self.page_head(page_id)
            global_head_sha256 = self.global_head()
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        return ProjectEditReadSnapshot(
            project=materialized,
            ledger=ledger,
            page_head_sha256=page_head_sha256,
            global_head_sha256=global_head_sha256,
        )

    def materialize_multi_page_snapshot(
        self,
        project: Mapping[str, Any],
    ) -> ProjectEditMultiPageReadSnapshot:
        """Read one materialized project, ledger, and all CAS heads atomically."""

        self._require_open()
        if self._connection.in_transaction:
            raise RuntimeError("project edit store already has an active transaction")
        self._connection.execute("BEGIN DEFERRED")
        try:
            materialized = self.materialize_project(project)
            ledger = ProjectEditLedger.from_dict(materialized["edit_ledger"])
            pages = materialized.get("pages")
            if not isinstance(pages, list):
                raise ValueError("project pages must be a list")
            page_ids = tuple(
                str(page.get("page_id") or "").strip()
                for page in pages
                if isinstance(page, Mapping)
            )
            if (
                len(page_ids) != len(pages)
                or any(not page_id for page_id in page_ids)
                or len(page_ids) != len(set(page_ids))
            ):
                raise ValueError("project page identities are invalid")
            page_heads = tuple(
                (page_id, self.page_head(page_id)) for page_id in sorted(page_ids)
            )
            global_head_sha256 = self.global_head()
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        return ProjectEditMultiPageReadSnapshot(
            project=materialized,
            ledger=ledger,
            page_head_sha256=page_heads,
            global_head_sha256=global_head_sha256,
        )

    def transaction_count(self) -> int:
        self._require_open()
        row = self._connection.execute(
            "SELECT COUNT(*) FROM edit_transactions"
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "ProjectEditStore":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
