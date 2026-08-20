# -*- coding: utf-8 -*-
"""Streaming canonical fingerprints for project-edit contracts."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


_GUI_ADDITIVE_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "edit_ledger",
        "artifact_revisions",
        "automated_state",
        "migration",
    }
)


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data without materializing one complete string."""

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


def automated_state_view(project: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow canonical view of immutable automated project state.

    Large page and bundle payloads remain referenced while the JSON encoder
    streams them into the digest.  GUI-owned additive sections and the schema
    marker are deliberately excluded so schema migration has a stable no-edit
    comparison.
    """

    if not isinstance(project, Mapping):
        raise TypeError("project must be a mapping")
    result = {
        str(key): value
        for key, value in project.items()
        if str(key) not in _GUI_ADDITIVE_ROOT_KEYS
    }
    metadata = result.get("project")
    if isinstance(metadata, Mapping) and "project_id" in metadata:
        metadata_view = dict(metadata)
        metadata_view.pop("project_id", None)
        result["project"] = metadata_view
    settings = result.get("settings")
    if isinstance(settings, Mapping):
        settings_view = dict(settings)
        for key in (
            "project_config",
            "provider_profile_refs",
            "last_run_snapshot",
        ):
            settings_view.pop(key, None)
        if settings_view:
            result["settings"] = settings_view
        else:
            result.pop("settings", None)
    return result


def automated_state_fingerprint(project: Mapping[str, Any]) -> str:
    return canonical_sha256(automated_state_view(project))


def project_id_for(
    project: Mapping[str, Any],
    *,
    stable_hint: str | None = None,
) -> str:
    metadata = project.get("project")
    if isinstance(metadata, Mapping):
        existing = str(metadata.get("project_id") or "").strip()
        if existing:
            return existing
    hint = str(stable_hint or "").strip()
    if hint:
        return f"project:run:{hint}"
    return f"project:{automated_state_fingerprint(project)[:32]}"


def project_origin_fingerprint(project: Mapping[str, Any]) -> str:
    """Fingerprint the stable project identity metadata, not page progress."""

    metadata = project.get("project")
    if not isinstance(metadata, Mapping):
        metadata = {}
    value = dict(metadata)
    value.pop("project_id", None)
    return canonical_sha256({"project": value})


def automatic_revision_id(value: Any, *, prefix: str) -> str:
    return f"{prefix}:{canonical_sha256(value)}"
