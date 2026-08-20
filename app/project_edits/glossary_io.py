"""Bounded JSON/CSV import and export for typed project glossary entries."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .glossary_commands import GlossaryEntryV1


MAX_GLOSSARY_FILE_BYTES = 2 * 1024 * 1024


def _generated_entry_id(source: str) -> str:
    digest = hashlib.sha256(source.strip().encode("utf-8")).hexdigest()[:20]
    return f"glossary-{digest}"


def _entry_from_import(value: Mapping[str, Any]) -> GlossaryEntryV1:
    data = dict(value)
    source = str(data.get("source") or "").strip()
    if not str(data.get("entry_id") or "").strip() and source:
        data["entry_id"] = _generated_entry_id(source)
    aliases = data.get("aliases", ())
    if isinstance(aliases, str):
        aliases_text = aliases.strip()
        if not aliases_text:
            aliases = ()
        elif aliases_text.startswith("["):
            decoded = json.loads(aliases_text)
            if not isinstance(decoded, list):
                raise ValueError("CSV aliases JSON must be a list")
            aliases = decoded
        else:
            aliases = [item.strip() for item in aliases_text.split("|") if item.strip()]
    data["aliases"] = aliases
    return GlossaryEntryV1.from_dict(data)


def parse_glossary_entries(data: bytes, *, suffix: str) -> tuple[GlossaryEntryV1, ...]:
    if not isinstance(data, bytes):
        raise TypeError("glossary import data must be bytes")
    if len(data) > MAX_GLOSSARY_FILE_BYTES:
        raise ValueError("glossary import exceeds the 2 MiB safety limit")
    text = data.decode("utf-8-sig")
    extension = suffix.lower()
    values: list[Mapping[str, Any]]
    if extension == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        expected = {"entry_id", "source", "target", "notes", "aliases", "priority"}
        if reader.fieldnames is None or set(reader.fieldnames) - expected:
            raise ValueError("glossary CSV has unsupported columns")
        values = [dict(row) for row in reader]
    elif extension == ".json":
        decoded = json.loads(text)
        if isinstance(decoded, Mapping):
            decoded = decoded.get("entries")
        if not isinstance(decoded, list):
            raise ValueError("glossary JSON must contain an entries list")
        if any(not isinstance(value, Mapping) for value in decoded):
            raise ValueError("glossary JSON entries must be objects")
        values = [dict(value) for value in decoded]
    else:
        raise ValueError("glossary files must use .json or .csv")
    entries = tuple(_entry_from_import(value) for value in values)
    if not entries:
        raise ValueError("glossary import contains no entries")
    if len({entry.entry_id for entry in entries}) != len(entries):
        raise ValueError("glossary import entry identities are duplicated")
    return entries


def load_glossary_entries(path: str) -> tuple[GlossaryEntryV1, ...]:
    source = Path(os.path.abspath(str(path or "")))
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if source.stat().st_size > MAX_GLOSSARY_FILE_BYTES:
        raise ValueError("glossary import exceeds the 2 MiB safety limit")
    return parse_glossary_entries(source.read_bytes(), suffix=source.suffix)


def serialize_glossary_entries(
    entries: Iterable[GlossaryEntryV1],
    *,
    suffix: str,
) -> bytes:
    values = tuple(entries)
    if any(not isinstance(entry, GlossaryEntryV1) for entry in values):
        raise TypeError("entries must contain GlossaryEntryV1 values")
    extension = suffix.lower()
    if extension == ".json":
        return (
            json.dumps(
                {"schema_version": "project_glossary_export_v1", "entries": [entry.to_dict() for entry in values]},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    if extension == ".csv":
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream,
            fieldnames=("entry_id", "source", "target", "notes", "aliases", "priority"),
            lineterminator="\n",
        )
        writer.writeheader()
        for entry in values:
            row = entry.to_dict()
            row["aliases"] = json.dumps(row["aliases"], ensure_ascii=False)
            writer.writerow(row)
        return stream.getvalue().encode("utf-8-sig")
    raise ValueError("glossary files must use .json or .csv")


def save_glossary_entries(path: str, entries: Iterable[GlossaryEntryV1]) -> str:
    target = Path(os.path.abspath(str(path or "")))
    payload = serialize_glossary_entries(entries, suffix=target.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(target)


__all__ = [
    "MAX_GLOSSARY_FILE_BYTES",
    "load_glossary_entries",
    "parse_glossary_entries",
    "save_glossary_entries",
    "serialize_glossary_entries",
]
