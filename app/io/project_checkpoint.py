# -*- coding: utf-8 -*-
"""Incremental durable project checkpoints for the single-page pipeline.

The store is deliberately persistence-only.  It records one already-complete
page and its already-produced style-context delta per transaction.  It does
not interpret pipeline, style, rendering, or translation contracts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Mapping
import uuid


PROJECT_CHECKPOINT_SCHEMA_VERSION = "project_checkpoint_v1"
PROJECT_CHECKPOINT_DESCRIPTOR_VERSION = "project_checkpoint_descriptor_v1"


@dataclass(frozen=True)
class ProjectCheckpointReceipt:
    sequence: int
    page_id: str
    page_sha256: str
    style_delta_sha256: str
    commit_sha256: str
    page_bytes: int
    style_delta_bytes: int
    encode_seconds: float
    transaction_seconds: float
    descriptor_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectFinalizeReceipt:
    page_count: int
    project_bytes: int
    project_sha256: str
    database_checkpoint_seconds: float
    recovery_seconds: float
    verification_seconds: float
    export_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_project_checkpoint_descriptor(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    checkpoint = value.get("checkpoint")
    return bool(
        isinstance(checkpoint, Mapping)
        and str(checkpoint.get("version") or "")
        == PROJECT_CHECKPOINT_DESCRIPTOR_VERSION
    )


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _decode_json(value: Any, *, field_name: str) -> Any:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{field_name} is not serialized JSON")
    return json.loads(value)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _commit_sha256(
    *,
    previous_commit_sha256: str,
    sequence: int,
    page_id: str,
    page_sha256: str,
    style_delta_sha256: str,
    style_cache_journal_id: str,
) -> str:
    payload = "\0".join(
        (
            PROJECT_CHECKPOINT_SCHEMA_VERSION,
            previous_commit_sha256,
            str(sequence),
            page_id,
            page_sha256,
            style_delta_sha256,
            style_cache_journal_id,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_store_path(json_path: str, store_basename: str) -> str:
    basename = str(store_basename or "")
    if not basename or basename != os.path.basename(basename):
        raise ValueError("checkpoint store path must be an adjacent basename")
    json_dir = os.path.dirname(os.path.abspath(json_path)) or os.getcwd()
    candidate = os.path.abspath(os.path.join(json_dir, basename))
    if os.path.dirname(candidate) != os.path.abspath(json_dir):
        raise ValueError("checkpoint store escaped the project directory")
    return candidate


def _connect(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _initialize_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA wal_autocheckpoint = 0")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS checkpoint_meta (
            key TEXT PRIMARY KEY,
            value BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS page_commits (
            sequence INTEGER PRIMARY KEY,
            page_id TEXT NOT NULL UNIQUE,
            page_payload BLOB NOT NULL,
            style_delta_payload BLOB,
            style_cache_journal_id TEXT NOT NULL,
            page_sha256 TEXT NOT NULL,
            style_delta_sha256 TEXT NOT NULL,
            previous_commit_sha256 TEXT NOT NULL,
            commit_sha256 TEXT NOT NULL UNIQUE,
            committed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS page_stage_outcomes (
            outcome_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            page_index INTEGER NOT NULL,
            outcome_payload BLOB NOT NULL,
            outcome_sha256 TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            UNIQUE(page_id, stage)
        );
        """
    )


def _read_meta(connection: sqlite3.Connection, key: str) -> Any:
    row = connection.execute(
        "SELECT value FROM checkpoint_meta WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        raise ValueError(f"checkpoint metadata is missing {key}")
    return _decode_json(row[0], field_name=f"checkpoint_meta.{key}")


def _apply_style_delta(
    style_context_cache: Any,
    style_delta: Any,
    *,
    journal_id: str,
) -> Any:
    if not isinstance(style_context_cache, Mapping):
        if style_delta is None:
            return style_context_cache
        raise ValueError("style delta exists without a style-context cache")
    cache = dict(style_context_cache)
    committed = list(cache.get("committed_deltas") or ())
    if style_delta is not None:
        if not isinstance(style_delta, Mapping):
            raise ValueError("style delta payload is not a mapping")
        page_identity = style_delta.get("page_identity")
        if not isinstance(page_identity, Mapping):
            raise ValueError("style delta page identity is missing")
        page_index = int(page_identity.get("page_index"))
        if page_index < 0 or page_index > len(committed):
            raise ValueError("style delta would create a cache-prefix gap")
        committed = [*committed[:page_index], dict(style_delta)]
    cache["committed_deltas"] = committed
    if journal_id:
        cache["journal_id"] = journal_id
    return cache


def _recover_project(connection: sqlite3.Connection) -> dict[str, Any]:
    schema_version = str(_read_meta(connection, "schema_version") or "")
    if schema_version != PROJECT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {schema_version}")
    base_project = _read_meta(connection, "base_project")
    if not isinstance(base_project, Mapping):
        raise ValueError("checkpoint base project is not a mapping")
    project = dict(base_project)
    pages: list[dict[str, Any]] = []
    style_context_cache = project.get("style_context_cache")
    previous_commit_sha256 = ""
    rows = connection.execute(
        """
        SELECT sequence, page_id, page_payload, style_delta_payload,
               style_cache_journal_id, page_sha256, style_delta_sha256,
               previous_commit_sha256, commit_sha256
        FROM page_commits
        ORDER BY sequence
        """
    ).fetchall()
    for expected_sequence, row in enumerate(rows):
        (
            sequence,
            page_id,
            page_payload,
            style_delta_payload,
            style_cache_journal_id,
            page_sha256,
            style_delta_sha256,
            stored_previous_sha256,
            commit_sha256,
        ) = row
        if int(sequence) != expected_sequence:
            raise ValueError("checkpoint page sequence is not contiguous")
        page_bytes = bytes(page_payload)
        delta_bytes = (
            bytes(style_delta_payload)
            if style_delta_payload is not None
            else b""
        )
        if _sha256(page_bytes) != str(page_sha256):
            raise ValueError(f"checkpoint page hash mismatch at {sequence}")
        if _sha256(delta_bytes) != str(style_delta_sha256):
            raise ValueError(f"checkpoint style-delta hash mismatch at {sequence}")
        if str(stored_previous_sha256) != previous_commit_sha256:
            raise ValueError(f"checkpoint chain mismatch at {sequence}")
        expected_commit_sha256 = _commit_sha256(
            previous_commit_sha256=previous_commit_sha256,
            sequence=int(sequence),
            page_id=str(page_id),
            page_sha256=str(page_sha256),
            style_delta_sha256=str(style_delta_sha256),
            style_cache_journal_id=str(style_cache_journal_id or ""),
        )
        if expected_commit_sha256 != str(commit_sha256):
            raise ValueError(f"checkpoint commit hash mismatch at {sequence}")
        page = _decode_json(page_bytes, field_name=f"page_commits[{sequence}]")
        if not isinstance(page, Mapping):
            raise ValueError(f"checkpoint page {sequence} is not a mapping")
        if str(page.get("page_id") or "") != str(page_id):
            raise ValueError(f"checkpoint page identity mismatch at {sequence}")
        delta = (
            _decode_json(
                delta_bytes,
                field_name=f"page_commits[{sequence}].style_delta",
            )
            if delta_bytes
            else None
        )
        style_context_cache = _apply_style_delta(
            style_context_cache,
            delta,
            journal_id=str(style_cache_journal_id or ""),
        )
        pages.append(dict(page))
        previous_commit_sha256 = str(commit_sha256)
    stage_outcomes = _recover_stage_outcomes(connection)
    completed_page_ids = {
        str(page.get("page_id") or "")
        for page in pages
        if str(page.get("page_id") or "")
    }
    outcomes_by_page: dict[str, list[dict[str, Any]]] = {}
    for outcome in stage_outcomes:
        page_id = str(outcome.get("page_id") or "")
        if page_id:
            outcomes_by_page.setdefault(page_id, []).append(outcome)
    for page_id, outcomes in sorted(
        outcomes_by_page.items(),
        key=lambda item: min(int(value.get("page_index") or 0) for value in item[1]),
    ):
        if page_id in completed_page_ids:
            continue
        failure = next(
            (
                outcome
                for outcome in reversed(outcomes)
                if str(outcome.get("state") or "") == "technical_failure"
            ),
            None,
        )
        latest = failure or outcomes[-1]
        source_path = str(latest.get("source_path") or "")
        page_name = str(latest.get("page_name") or os.path.basename(source_path) or page_id)
        artifact_summary = _merge_valid_stage_artifacts(outcomes)
        pages.append(
            {
                "page_id": page_id,
                "file_name": page_name,
                "image_path": source_path,
                "output_path": str(artifact_summary.get("output_path") or ""),
                "width": int(artifact_summary.get("width") or 0),
                "height": int(artifact_summary.get("height") or 0),
                "page_class": str(artifact_summary.get("page_class") or "unknown"),
                "regions": list(artifact_summary.get("regions") or []),
                "parent_execution_bundles": list(
                    artifact_summary.get("parent_execution_bundles") or []
                ),
                "text_area_plan": artifact_summary.get("text_area_plan"),
                "cleaned_page_base": dict(
                    artifact_summary.get("cleaned_page_base") or {}
                ),
                "processing_state": "failed" if failure is not None else "in_progress",
                "failed_stage": str((failure or {}).get("stage") or ""),
                "pipeline_error": dict(failure or {}),
            }
        )
    pages.sort(
        key=lambda page: min(
            [
                int(value.get("page_index") or 0)
                for value in outcomes_by_page.get(str(page.get("page_id") or ""), [])
            ]
            or [len(pages)]
        )
    )
    project["pages"] = pages
    if stage_outcomes:
        project["stage_outcomes"] = stage_outcomes
    if style_context_cache is not None:
        project["style_context_cache"] = style_context_cache
    return project


def _merge_valid_stage_artifacts(
    outcomes: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recover the latest durable artifact from every successful stage."""

    merged: dict[str, Any] = {}
    for outcome in outcomes:
        if str(outcome.get("state") or "") not in {
            "valid",
            "valid_with_diagnostics",
        }:
            continue
        summary = outcome.get("artifact_summary")
        if not isinstance(summary, Mapping):
            continue
        for key in (
            "width",
            "height",
            "page_class",
            "text_area_plan",
            "cleaned_page_base",
            "output_path",
        ):
            value = summary.get(key)
            if value not in (None, "", [], {}):
                merged[key] = value
        for key in ("regions", "parent_execution_bundles"):
            value = summary.get(key)
            if isinstance(value, list) and value:
                merged[key] = value
    return merged


def _recover_stage_outcomes(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    try:
        rows = connection.execute(
            "SELECT outcome_payload, outcome_sha256 FROM page_stage_outcomes "
            "ORDER BY page_index, outcome_sequence"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    outcomes: list[dict[str, Any]] = []
    for index, (payload, expected_sha256) in enumerate(rows):
        payload_bytes = bytes(payload)
        if _sha256(payload_bytes) != str(expected_sha256):
            raise ValueError(f"stage outcome hash mismatch at {index}")
        value = _decode_json(payload_bytes, field_name=f"page_stage_outcomes[{index}]")
        if not isinstance(value, Mapping):
            raise ValueError(f"stage outcome {index} is not a mapping")
        outcomes.append(dict(value))
    return outcomes


def recover_project_from_descriptor(
    json_path: str,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = descriptor.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("project checkpoint descriptor is missing")
    if (
        str(checkpoint.get("version") or "")
        != PROJECT_CHECKPOINT_DESCRIPTOR_VERSION
    ):
        raise ValueError("unsupported project checkpoint descriptor")
    store_path = _safe_store_path(
        json_path,
        str(checkpoint.get("store") or ""),
    )
    if not os.path.isfile(store_path):
        raise FileNotFoundError(f"project checkpoint store is missing: {store_path}")
    connection = _connect(store_path)
    try:
        run_id = str(_read_meta(connection, "run_id") or "")
        if run_id != str(checkpoint.get("run_id") or ""):
            raise ValueError("project checkpoint run identity mismatch")
        return _recover_project(connection)
    finally:
        connection.close()


class ProjectCheckpointSession:
    """One controller-owned forward-only checkpoint session."""

    def __init__(
        self,
        *,
        json_path: str,
        base_project: Mapping[str, Any],
        run_id: str | None = None,
    ) -> None:
        self.json_path = os.path.abspath(json_path)
        json_dir = os.path.dirname(self.json_path) or os.getcwd()
        os.makedirs(json_dir, exist_ok=True)
        self.run_id = str(run_id or uuid.uuid4().hex)
        if not self.run_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in self.run_id
        ):
            raise ValueError("checkpoint run identity is not path-safe")
        self.store_basename = (
            f".{os.path.basename(self.json_path)}."
            f"{self.run_id[:24]}.checkpoint.sqlite3"
        )
        self.store_path = _safe_store_path(
            self.json_path,
            self.store_basename,
        )
        base = dict(base_project or {})
        base["pages"] = []
        self._connection = _connect(self.store_path)
        _initialize_connection(self._connection)
        self._descriptor_published = False
        self._descriptor_payload: bytes | None = None
        self._closed = False
        self._receipts: list[ProjectCheckpointReceipt] = []
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for key, value in (
                ("schema_version", PROJECT_CHECKPOINT_SCHEMA_VERSION),
                ("run_id", self.run_id),
                ("created_at", _utc_now()),
                ("base_project", base),
            ):
                self._connection.execute(
                    "INSERT INTO checkpoint_meta(key, value) VALUES (?, ?)",
                    (key, _compact_json_bytes(value)),
                )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            self._connection.close()
            raise
        self._publish_descriptor()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("project checkpoint session is closed")

    def _descriptor(self) -> dict[str, Any]:
        base_project = _read_meta(self._connection, "base_project")
        return {
            "schema_version": str(base_project.get("schema_version") or "1.0"),
            "project": dict(base_project.get("project") or {}),
            "pages": [],
            "checkpoint": {
                "version": PROJECT_CHECKPOINT_DESCRIPTOR_VERSION,
                "run_id": self.run_id,
                "store": self.store_basename,
            },
        }

    def _descriptor_is_current(self) -> bool:
        payload = self._descriptor_payload
        if not self._descriptor_published or payload is None:
            return False
        try:
            if os.path.getsize(self.json_path) != len(payload):
                return False
            with open(self.json_path, "rb") as handle:
                return handle.read(len(payload) + 1) == payload
        except OSError:
            return False

    def _publish_descriptor(self) -> None:
        from app.io.project import save_project_atomic

        descriptor = self._descriptor()
        save_project_atomic(
            self.json_path,
            descriptor,
            compact=True,
        )
        self._descriptor_payload = _compact_json_bytes(descriptor)
        self._descriptor_published = True

    def commit_page(
        self,
        *,
        page_record: Mapping[str, Any],
        style_delta: Mapping[str, Any] | None,
        style_cache_journal_id: str,
    ) -> ProjectCheckpointReceipt:
        self._require_open()
        total_started = time.perf_counter()
        encode_started = time.perf_counter()
        page_id = str(page_record.get("page_id") or "")
        if not page_id:
            raise ValueError("checkpoint page identity is empty")
        page_payload = _compact_json_bytes(dict(page_record))
        style_delta_payload = (
            _compact_json_bytes(dict(style_delta))
            if style_delta is not None
            else b""
        )
        page_sha256 = _sha256(page_payload)
        style_delta_sha256 = _sha256(style_delta_payload)
        encode_seconds = time.perf_counter() - encode_started

        last = self._connection.execute(
            "SELECT sequence, commit_sha256 FROM page_commits "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = int(last[0]) + 1 if last is not None else 0
        previous_commit_sha256 = str(last[1]) if last is not None else ""
        commit_sha256 = _commit_sha256(
            previous_commit_sha256=previous_commit_sha256,
            sequence=sequence,
            page_id=page_id,
            page_sha256=page_sha256,
            style_delta_sha256=style_delta_sha256,
            style_cache_journal_id=str(style_cache_journal_id or ""),
        )
        transaction_started = time.perf_counter()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute(
                """
                INSERT INTO page_commits(
                    sequence, page_id, page_payload, style_delta_payload,
                    style_cache_journal_id, page_sha256, style_delta_sha256,
                    previous_commit_sha256, commit_sha256, committed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sequence,
                    page_id,
                    page_payload,
                    style_delta_payload or None,
                    str(style_cache_journal_id or ""),
                    page_sha256,
                    style_delta_sha256,
                    previous_commit_sha256,
                    commit_sha256,
                    _utc_now(),
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        transaction_seconds = time.perf_counter() - transaction_started

        descriptor_started = time.perf_counter()
        if not self._descriptor_is_current():
            # Legacy review tools may materialize and save the hydrated project
            # while a run is active.  The next durable page commit restores the
            # checkpoint link without changing those tools or pipeline order.
            self._publish_descriptor()
        descriptor_seconds = time.perf_counter() - descriptor_started

        receipt = ProjectCheckpointReceipt(
            sequence=sequence,
            page_id=page_id,
            page_sha256=page_sha256,
            style_delta_sha256=style_delta_sha256,
            commit_sha256=commit_sha256,
            page_bytes=len(page_payload),
            style_delta_bytes=len(style_delta_payload),
            encode_seconds=encode_seconds,
            transaction_seconds=transaction_seconds,
            descriptor_seconds=descriptor_seconds,
            total_seconds=time.perf_counter() - total_started,
        )
        self._receipts.append(receipt)
        return receipt

    def record_stage_outcome(
        self,
        outcome: Mapping[str, Any],
    ) -> None:
        """Durably publish one latest owner-stage outcome before page commit."""

        self._require_open()
        if not isinstance(outcome, Mapping):
            raise TypeError("stage outcome must be mapping-like")
        page_id = str(outcome.get("page_id") or "")
        stage = str(outcome.get("stage") or "")
        state = str(outcome.get("state") or "")
        if not page_id or not stage:
            raise ValueError("stage outcome requires page_id and stage")
        if state not in {"valid", "valid_with_diagnostics", "technical_failure"}:
            raise ValueError("stage outcome state is invalid")
        page_index = int(outcome.get("page_index") or 0)
        payload = _compact_json_bytes(dict(outcome))
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute(
                """
                INSERT INTO page_stage_outcomes(
                    page_id, stage, page_index, outcome_payload,
                    outcome_sha256, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(page_id, stage) DO UPDATE SET
                    page_index=excluded.page_index,
                    outcome_payload=excluded.outcome_payload,
                    outcome_sha256=excluded.outcome_sha256,
                    recorded_at=excluded.recorded_at
                """,
                (
                    page_id,
                    stage,
                    page_index,
                    payload,
                    _sha256(payload),
                    _utc_now(),
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        if not self._descriptor_is_current():
            self._publish_descriptor()

    def stage_outcomes(self) -> list[dict[str, Any]]:
        self._require_open()
        return _recover_stage_outcomes(self._connection)

    def recover_project(self) -> dict[str, Any]:
        self._require_open()
        return _recover_project(self._connection)

    def finalize(
        self,
        *,
        expected_project: Mapping[str, Any] | None = None,
    ) -> ProjectFinalizeReceipt:
        self._require_open()
        if not self._descriptor_published:
            raise ValueError("cannot finalize a project without a committed page")
        from app.io.project import save_project_bytes_atomic

        total_started = time.perf_counter()
        database_checkpoint_started = time.perf_counter()
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        database_checkpoint_seconds = (
            time.perf_counter() - database_checkpoint_started
        )
        recovery_started = time.perf_counter()
        recovered = self.recover_project()
        recovery_seconds = time.perf_counter() - recovery_started
        verification_started = time.perf_counter()
        recovered_payload = _compact_json_bytes(recovered)
        expected = dict(expected_project) if expected_project is not None else None
        if expected is not None and "stage_outcomes" in recovered:
            expected["stage_outcomes"] = list(recovered.get("stage_outcomes") or [])
        if expected is not None and recovered != expected:
            raise ValueError(
                "incremental checkpoint does not match the expected project"
            )
        verification_seconds = time.perf_counter() - verification_started
        export_started = time.perf_counter()
        save_project_bytes_atomic(self.json_path, recovered_payload)
        export_seconds = time.perf_counter() - export_started
        return ProjectFinalizeReceipt(
            page_count=len(recovered.get("pages") or ()),
            project_bytes=len(recovered_payload),
            project_sha256=_sha256(recovered_payload),
            database_checkpoint_seconds=database_checkpoint_seconds,
            recovery_seconds=recovery_seconds,
            verification_seconds=verification_seconds,
            export_seconds=export_seconds,
            total_seconds=time.perf_counter() - total_started,
        )

    def summary(self) -> dict[str, Any]:
        receipts = tuple(self._receipts)
        return {
            "version": PROJECT_CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "store": self.store_path,
            "commit_count": len(receipts),
            "page_bytes": sum(item.page_bytes for item in receipts),
            "style_delta_bytes": sum(
                item.style_delta_bytes for item in receipts
            ),
            "encode_seconds": sum(item.encode_seconds for item in receipts),
            "transaction_seconds": sum(
                item.transaction_seconds for item in receipts
            ),
            "descriptor_seconds": sum(
                item.descriptor_seconds for item in receipts
            ),
            "total_seconds": sum(item.total_seconds for item in receipts),
            "commits": [item.to_dict() for item in receipts],
            "stage_outcome_count": len(_recover_stage_outcomes(self._connection)),
        }

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "ProjectCheckpointSession":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
