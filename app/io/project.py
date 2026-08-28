# -*- coding: utf-8 -*-
"""Project IO helpers."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Iterator, Mapping, Sequence

if TYPE_CHECKING:
    from app.config.settings_contracts import (
        ModuleConfig,
        ProjectConfig,
        RunSettingsSnapshot,
    )


PROJECT_SCHEMA_V1 = "1.0"
PROJECT_SCHEMA_V2 = "2.0"


def _acquire_project_publication_lock(lock_stream: Any) -> None:
    lock_stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_stream.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)


def _release_project_publication_lock(lock_stream: Any) -> None:
    lock_stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class ProjectSettingsState:
    """Typed GUI-owned settings persisted beside immutable project output."""

    project_config: ProjectConfig
    module_configs: tuple[ModuleConfig, ...] = ()
    last_run_snapshot: RunSettingsSnapshot | None = None


def default_project_dict() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "project": {
            "name": "",
            "language": {"source": "ja", "target": "zh-Hans"},
            "created_at": "",
            "model": {"detector": "ComicTextDetector", "ocr": "PaddleOCR-VL", "translator": "ollama:auto"},
            "style_guide": "",
        },
        "pages": [],
    }


@contextmanager
def _project_publication_lock(path: str) -> Iterator[None]:
    """Serialize every writer that can publish the project JSON path.

    The forward checkpoint descriptor and explicit schema-2 export share this
    lock.  Holding it across the descriptor check and atomic replace closes the
    window where an editor export could overwrite a newly activated run.
    """

    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path) or os.getcwd()
    os.makedirs(parent, exist_ok=True)
    lock_path = os.path.join(
        parent,
        f".{os.path.basename(absolute_path)}.publish.lock",
    )
    with open(lock_path, "a+b") as lock_stream:
        lock_stream.seek(0, os.SEEK_END)
        if lock_stream.tell() == 0:
            lock_stream.write(b"\0")
            lock_stream.flush()
        _acquire_project_publication_lock(lock_stream)
        try:
            yield
        finally:
            _release_project_publication_lock(lock_stream)


def save_project(path: str, data: Dict[str, Any]) -> None:
    with _project_publication_lock(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _save_project_atomic_unlocked(
    path: str,
    data: Mapping[str, Any],
    *,
    compact: bool,
) -> None:
    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path) or os.getcwd()
    os.makedirs(parent, exist_ok=True)
    handle = -1
    temp_path = ""
    try:
        handle, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(absolute_path)}.",
            suffix=".tmp",
            dir=parent,
        )
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            handle = -1
            if compact:
                json.dump(
                    data,
                    stream,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            else:
                json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, absolute_path)
        temp_path = ""
    finally:
        if handle >= 0:
            os.close(handle)
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def save_project_atomic(
    path: str,
    data: Dict[str, Any],
    *,
    compact: bool = False,
) -> None:
    """Write one complete project view without exposing a partial JSON file."""
    with _project_publication_lock(path):
        _save_project_atomic_unlocked(path, data, compact=compact)


def _save_project_bytes_atomic_unlocked(path: str, payload: bytes) -> None:
    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path) or os.getcwd()
    os.makedirs(parent, exist_ok=True)
    handle = -1
    temp_path = ""
    try:
        handle, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(absolute_path)}.",
            suffix=".tmp",
            dir=parent,
        )
        with os.fdopen(handle, "wb") as stream:
            handle = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, absolute_path)
        temp_path = ""
    finally:
        if handle >= 0:
            os.close(handle)
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def save_project_bytes_atomic(path: str, payload: bytes) -> None:
    """Atomically publish an already-serialized complete project payload."""

    if not isinstance(payload, bytes):
        raise TypeError("project payload must be bytes")
    with _project_publication_lock(path):
        _save_project_bytes_atomic_unlocked(path, payload)


def load_project(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        project = json.load(f)
    from app.io.project_checkpoint import (
        is_project_checkpoint_descriptor,
        recover_project_from_descriptor,
    )

    if is_project_checkpoint_descriptor(project):
        return recover_project_from_descriptor(path, project)
    return project


def project_storage_is_checkpoint_descriptor(path: str) -> bool:
    """Return whether the on-disk project entry is a forward checkpoint.

    ``load_project`` intentionally recovers checkpoint descriptors into their
    materialized project view.  GUI persistence needs this separate probe so
    it can defer settings publication instead of attempting to replace the
    controller-owned descriptor.
    """

    with open(path, "r", encoding="utf-8") as stream:
        stored = json.load(stream)
    from app.io.project_checkpoint import is_project_checkpoint_descriptor

    return is_project_checkpoint_descriptor(stored)


def _artifact_revision_id(kind: str, page_id: str, descriptor: Mapping[str, Any]) -> str:
    from app.project_edits.fingerprints import canonical_sha256

    return f"{kind}:{page_id}:{canonical_sha256(descriptor)[:32]}"


def _catalog_artifact_revisions(project: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cleaned_page_bases: list[dict[str, Any]] = []
    rendered_pages: list[dict[str, Any]] = []
    parent_layers: list[dict[str, Any]] = []
    source_revisions: list[dict[str, Any]] = []
    translation_revisions: list[dict[str, Any]] = []
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise ValueError("project pages must be a list")
    for page in pages:
        if not isinstance(page, Mapping):
            raise ValueError("project page records must be mappings")
        page_id = str(page.get("page_id") or "").strip()
        if not page_id:
            raise ValueError("project page identity is missing")
        cleaned = page.get("cleaned_page_base")
        if isinstance(cleaned, Mapping):
            descriptor = {
                "state": str(cleaned.get("state") or ""),
                "valid": bool(cleaned.get("valid")),
                "asset": str(
                    cleaned.get("image_path")
                    or cleaned.get("cache_path")
                    or ""
                ),
                "content_sha256": str(
                    cleaned.get("cleaned_page_base_sha256")
                    or cleaned.get("source_sha256")
                    or ""
                ),
                "cleaned_page_base": dict(cleaned),
            }
            cleaned_page_bases.append(
                {
                    "revision_id": _artifact_revision_id(
                        "cleaned", page_id, descriptor
                    ),
                    "page_id": page_id,
                    "provenance": "automatic",
                    "current": True,
                    **descriptor,
                }
            )
        output_path = str(page.get("output_path") or "")
        if output_path:
            descriptor = {
                "asset": output_path,
                "content_sha256": str(
                    page.get("output_sha256")
                    or page.get("rendered_page_sha256")
                    or ""
                ),
            }
            rendered_pages.append(
                {
                    "revision_id": _artifact_revision_id(
                        "rendered", page_id, descriptor
                    ),
                    "page_id": page_id,
                    "provenance": "automatic",
                    "current": True,
                    **descriptor,
                }
            )
        existing_layers = page.get("parent_layers") or ()
        if existing_layers and not isinstance(existing_layers, (list, tuple)):
            raise ValueError("parent layer artifacts must be a list")
        for layer in existing_layers:
            if not isinstance(layer, Mapping):
                raise ValueError("parent layer artifacts must be mappings")
            descriptor = dict(layer)
            parent_id = str(
                descriptor.get("parent_id")
                or descriptor.get("bundle_id")
                or ""
            )
            parent_layers.append(
                {
                    "revision_id": _artifact_revision_id(
                        "parent-layer", page_id, descriptor
                    ),
                    "page_id": page_id,
                    "parent_id": parent_id,
                    "provenance": "automatic",
                    "artifact": descriptor,
                }
            )
    return {
        "cleaned_page_bases": cleaned_page_bases,
        "rendered_pages": rendered_pages,
        "parent_layers": parent_layers,
        "source_revisions": source_revisions,
        "translation_revisions": translation_revisions,
    }


def _migration_warnings(project: Mapping[str, Any]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        return warnings
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        page_id = str(page.get("page_id") or "")
        regions = page.get("regions") or ()
        if not isinstance(regions, (list, tuple)):
            continue
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            flags = region.get("flags")
            if isinstance(flags, Mapping) and any(
                bool(flags.get(name)) for name in ("ignore", "needs_review")
            ):
                warnings.append(
                    {
                        "page_id": page_id,
                        "region_id": str(region.get("region_id") or ""),
                        "reason": "legacy_region_review_flags_retained_as_evidence",
                    }
                )
    return warnings


def migrate_project_schema_v2(
    project: Mapping[str, Any],
    *,
    project_id: str | None = None,
) -> Dict[str, Any]:
    """Build an additive schema-2 project view without mutating its input.

    This is an explicit editing/migration API.  The legacy ``load_project``
    path intentionally keeps returning the stored schema until the native GUI
    migration surface is introduced.
    """

    if not isinstance(project, Mapping):
        raise TypeError("project must be a mapping")
    schema_version = str(project.get("schema_version") or PROJECT_SCHEMA_V1)
    if schema_version == PROJECT_SCHEMA_V2:
        validate_project_schema_v2(project)
        existing_id = str(
            (project.get("project") or {}).get("project_id") or ""
        )
        if project_id and existing_id != str(project_id):
            raise ValueError("schema-2 project identity does not match requested identity")
        return dict(project)
    if schema_version != PROJECT_SCHEMA_V1:
        raise ValueError(f"unsupported project schema: {schema_version}")

    from app.project_edits.contracts import LEDGER_SCHEMA_VERSION
    from app.project_edits.fingerprints import (
        automated_state_fingerprint,
        project_id_for,
    )

    automated_sha256 = automated_state_fingerprint(project)
    migrated = dict(project)
    metadata = dict(project.get("project") or {})
    resolved_project_id = str(project_id or "").strip() or project_id_for(project)
    metadata["project_id"] = resolved_project_id
    migrated["project"] = metadata
    pages = project.get("pages") or []
    if not isinstance(pages, list):
        raise ValueError("project pages must be a list")
    migrated["pages"] = list(pages)
    existing_settings = project.get("settings")
    settings = dict(existing_settings) if isinstance(existing_settings, Mapping) else {}
    settings.setdefault("project_config", {})
    settings.setdefault("provider_profile_refs", {})
    settings.setdefault("last_run_snapshot", {})
    migrated["settings"] = settings
    migrated["edit_ledger"] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "project_id": metadata["project_id"],
        "edits": [],
    }
    migrated["artifact_revisions"] = _catalog_artifact_revisions(project)
    migrated["automated_state"] = {
        "schema_version": "automated_state_fingerprint_v1",
        "sha256": automated_sha256,
    }
    migrated["migration"] = {
        "source_schema_version": PROJECT_SCHEMA_V1,
        "warnings": _migration_warnings(project),
    }
    migrated["schema_version"] = PROJECT_SCHEMA_V2
    validate_project_schema_v2(migrated)
    if automated_state_fingerprint(migrated) != automated_sha256:
        raise ValueError("schema migration changed immutable automated state")
    return migrated


def validate_project_schema_v2(project: Mapping[str, Any]) -> None:
    if str(project.get("schema_version") or "") != PROJECT_SCHEMA_V2:
        raise ValueError("schema-2 project is required")
    metadata = project.get("project")
    if not isinstance(metadata, Mapping) or not str(metadata.get("project_id") or ""):
        raise ValueError("schema-2 project identity is missing")
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise ValueError("schema-2 pages must be a list")
    settings = project.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("schema-2 settings section is missing")
    ledger = project.get("edit_ledger")
    if not isinstance(ledger, Mapping):
        raise ValueError("schema-2 edit ledger is missing")
    from app.project_edits.ledger import ProjectEditLedger

    parsed_ledger = ProjectEditLedger.from_dict(ledger)
    if parsed_ledger.project_id != str(metadata.get("project_id") or ""):
        raise ValueError("schema-2 edit ledger project identity mismatch")
    artifacts = project.get("artifact_revisions")
    if not isinstance(artifacts, Mapping):
        raise ValueError("schema-2 artifact revision catalog is missing")
    for key in ("cleaned_page_bases", "rendered_pages", "parent_layers"):
        if not isinstance(artifacts.get(key), list):
            raise ValueError(f"schema-2 artifact catalog {key} must be a list")
    if "source_revisions" in artifacts and not isinstance(
        artifacts.get("source_revisions"), list
    ):
        raise ValueError("schema-2 artifact catalog source_revisions must be a list")
    if "translation_revisions" in artifacts and not isinstance(
        artifacts.get("translation_revisions"), list
    ):
        raise ValueError(
            "schema-2 artifact catalog translation_revisions must be a list"
        )
    state = project.get("automated_state")
    if not isinstance(state, Mapping):
        raise ValueError("schema-2 automated-state fingerprint is missing")
    from app.project_edits.fingerprints import (
        automated_state_fingerprint,
        project_id_for,
        project_origin_fingerprint,
    )

    expected = str(state.get("sha256") or "")
    if str(state.get("schema_version") or "") != "automated_state_fingerprint_v1":
        raise ValueError("schema-2 automated-state fingerprint schema is invalid")
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("schema-2 automated-state fingerprint is invalid")
    if expected != automated_state_fingerprint(project):
        raise ValueError("schema-2 automated-state fingerprint mismatch")


def _validate_project_module_configs(
    module_configs: Sequence[ModuleConfig],
) -> tuple[ModuleConfig, ...]:
    from app.config.module_registry import DEFAULT_MODULE_REGISTRY
    from app.config.settings_contracts import ModuleConfig, SettingsScope

    if isinstance(module_configs, (str, bytes, bytearray)) or not isinstance(
        module_configs, Sequence
    ):
        raise TypeError("project module configs must be a sequence")
    configs = tuple(module_configs)
    if any(not isinstance(config, ModuleConfig) for config in configs):
        raise TypeError("project module configs must contain ModuleConfig values")
    module_ids = [config.module_id for config in configs]
    if len(module_ids) != len(set(module_ids)):
        raise ValueError("project module configs contain duplicate module IDs")
    for config in configs:
        module = DEFAULT_MODULE_REGISTRY.get_module(config.module_id)
        DEFAULT_MODULE_REGISTRY.validate_config(config, allow_legacy=True)
        for collection_name, values in (
            ("values", config.values),
            ("legacy_values", config.legacy_values),
        ):
            for setting_id in values:
                definition = module.definitions.get(setting_id)
                if definition is None or definition.scope is not SettingsScope.PROJECT:
                    raise ValueError(
                        f"{config.module_id}.{setting_id} cannot be persisted as "
                        f"project-scoped {collection_name}"
                    )
    return tuple(sorted(configs, key=lambda config: config.module_id))


def read_project_settings(project: Mapping[str, Any]) -> ProjectSettingsState:
    """Decode the GUI-owned project settings without parsing them in widgets."""

    validate_project_schema_v2(project)
    from app.config.settings_contracts import (
        ProjectConfig,
        module_config_from_dict,
        project_config_from_dict,
        run_settings_snapshot_from_dict,
    )

    settings = project["settings"]
    raw_container = settings.get("project_config") or {}
    if not isinstance(raw_container, Mapping):
        raise ValueError("project settings project_config must be a mapping")
    if raw_container:
        if frozenset(raw_container) != {"project", "module_configs"}:
            raise ValueError("project_config container has unsupported fields")
        raw_project = raw_container["project"]
        raw_modules = raw_container["module_configs"]
        if not isinstance(raw_project, Mapping):
            raise ValueError("project_config.project must be a mapping")
        if not isinstance(raw_modules, list):
            raise ValueError("project_config.module_configs must be a list")
        project_config = project_config_from_dict(raw_project)
        module_configs = _validate_project_module_configs(
            tuple(module_config_from_dict(item) for item in raw_modules)
        )
    else:
        project_config = ProjectConfig()
        module_configs = ()

    raw_references = settings.get("provider_profile_refs") or {}
    if not isinstance(raw_references, Mapping):
        raise ValueError("provider_profile_refs must be a mapping")
    if dict(raw_references) != dict(project_config.provider_profile_references):
        raise ValueError("provider profile references disagree with ProjectConfig")

    raw_snapshot = settings.get("last_run_snapshot") or {}
    if not isinstance(raw_snapshot, Mapping):
        raise ValueError("last_run_snapshot must be a mapping")
    snapshot = (
        run_settings_snapshot_from_dict(raw_snapshot) if raw_snapshot else None
    )
    project_id = str((project.get("project") or {}).get("project_id") or "")
    if snapshot is not None and snapshot.project_id != project_id:
        raise ValueError("last run snapshot belongs to another project")
    return ProjectSettingsState(
        project_config=project_config,
        module_configs=module_configs,
        last_run_snapshot=snapshot,
    )


def with_project_settings(
    project: Mapping[str, Any],
    *,
    project_config: ProjectConfig,
    module_configs: Sequence[ModuleConfig],
    last_run_snapshot: RunSettingsSnapshot | None,
) -> Dict[str, Any]:
    """Return a schema-2 project view with typed GUI settings replaced."""

    validate_project_schema_v2(project)
    from app.config.settings_contracts import (
        ProjectConfig,
        RunSettingsSnapshot,
    )
    from app.project_edits.fingerprints import automated_state_fingerprint

    if not isinstance(project_config, ProjectConfig):
        raise TypeError("project_config must be ProjectConfig")
    configs = _validate_project_module_configs(module_configs)
    if last_run_snapshot is not None and not isinstance(
        last_run_snapshot, RunSettingsSnapshot
    ):
        raise TypeError("last_run_snapshot must be RunSettingsSnapshot or None")
    project_id = str((project.get("project") or {}).get("project_id") or "")
    if last_run_snapshot is not None and last_run_snapshot.project_id != project_id:
        raise ValueError("last run snapshot belongs to another project")

    before = automated_state_fingerprint(project)
    updated = dict(project)
    settings = dict(project.get("settings") or {})
    settings["project_config"] = {
        "project": project_config.to_dict(),
        "module_configs": [config.to_dict() for config in configs],
    }
    settings["provider_profile_refs"] = dict(
        project_config.provider_profile_references
    )
    settings["last_run_snapshot"] = (
        last_run_snapshot.to_dict() if last_run_snapshot is not None else {}
    )
    updated["settings"] = settings
    validate_project_schema_v2(updated)
    if automated_state_fingerprint(updated) != before:
        raise ValueError("GUI settings update changed immutable automated state")
    read_project_settings(updated)
    return updated


def load_project_for_editing(path: str) -> Dict[str, Any]:
    """Recover the automated base, then hydrate its adjacent edit journal.

    Merely opening a project never creates a sidecar and never republishes the
    controller-owned checkpoint descriptor.
    """

    from app.io.project_checkpoint import (
        is_project_checkpoint_descriptor,
        recover_project_from_descriptor,
    )
    from app.io.project_edit_store import (
        ProjectEditStore,
        inspect_project_edit_store,
    )
    from app.project_edits.fingerprints import (
        automated_state_fingerprint,
        project_id_for,
        project_origin_fingerprint,
    )

    with open(path, "r", encoding="utf-8") as stream:
        stored = json.load(stream)
    descriptor = stored if is_project_checkpoint_descriptor(stored) else None
    recovered = (
        recover_project_from_descriptor(path, stored)
        if descriptor is not None
        else stored
    )
    store_metadata = inspect_project_edit_store(path)
    if descriptor is not None:
        checkpoint = descriptor.get("checkpoint") or {}
        run_id = str(checkpoint.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("project checkpoint run identity is missing")
        explicit_project_id = f"project:run:{run_id}"
    else:
        explicit_project_id = project_id_for(recovered)
        if (
            store_metadata is not None
            and str(store_metadata.get("project_id") or "").startswith(
                "project:run:"
            )
            and not str(
                ((recovered.get("project") or {}).get("project_id") or "")
            )
        ):
            stored_project_id = str(store_metadata["project_id"])
            run_id = stored_project_id.removeprefix("project:run:")
            checkpoint_store = os.path.join(
                os.path.dirname(os.path.abspath(path)) or os.getcwd(),
                f".{os.path.basename(path)}.{run_id[:24]}.checkpoint.sqlite3",
            )
            if not os.path.isfile(checkpoint_store):
                raise ValueError(
                    "run-bound edit store has no matching checkpoint identity"
                )
            explicit_project_id = stored_project_id
    origin_sha256 = project_origin_fingerprint(recovered)
    if store_metadata is not None:
        if str(store_metadata.get("project_id") or "") != explicit_project_id:
            raise ValueError("project and edit-store identities do not match")
        if str(store_metadata.get("project_origin_sha256") or "") != origin_sha256:
            raise ValueError("project and edit-store origins do not match")
    migrated = migrate_project_schema_v2(
        recovered,
        project_id=explicit_project_id or None,
    )
    if store_metadata is None:
        return migrated
    project_id = str((migrated.get("project") or {}).get("project_id") or "")
    from app.project_edits.ledger import ProjectEditLedger

    embedded_ledger = ProjectEditLedger.from_dict(migrated["edit_ledger"])
    with ProjectEditStore(
        project_path=path,
        project_id=project_id,
        project_origin_sha256=origin_sha256,
        automated_state_sha256=automated_state_fingerprint(migrated),
        base_ledger=embedded_ledger,
        base_artifact_revisions=migrated["artifact_revisions"],
    ) as store:
        return store.materialize_project(migrated)


def save_project_schema_v2_atomic(
    path: str,
    project: Mapping[str, Any],
    *,
    defer_if_checkpoint: bool = False,
) -> bool:
    """Explicitly publish a validated schema-2 project view when idle.

    A live checkpoint descriptor is controller-owned.  Replacing it, even via
    an atomic rename, would be lost on the controller's next page commit.  The
    optional deferred mode returns ``False`` under the same publication lock;
    normal callers retain the historical exception and successful writes
    return ``True``.
    """

    if type(defer_if_checkpoint) is not bool:
        raise TypeError("defer_if_checkpoint must be a bool")
    validate_project_schema_v2(project)
    with _project_publication_lock(path):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as stream:
                current = json.load(stream)
            from app.io.project_checkpoint import is_project_checkpoint_descriptor

            if is_project_checkpoint_descriptor(current):
                if defer_if_checkpoint:
                    return False
                raise RuntimeError(
                    "schema-2 materialization is unavailable while a forward "
                    "checkpoint is active"
                )
        _save_project_atomic_unlocked(path, dict(project), compact=False)
    return True
