# -*- coding: utf-8 -*-
"""Forward-only transport for qualified prior-page source-style evidence."""
from __future__ import annotations

import hashlib
import json
import ntpath
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping


STYLE_CONTEXT_CACHE_VERSION = "style_context_cache_v1"
STYLE_CONTEXT_POLICY_VERSION = "style_context_policy_v1"
STYLE_CONTEXT_RUN_IDENTITY_VERSION = "style_context_run_identity_v1"
STYLE_CONTEXT_PAGE_IDENTITY_VERSION = "style_context_page_identity_v1"
STYLE_CONTEXT_DELTA_VERSION = "style_context_delta_v1"
STYLE_CONTEXT_SNAPSHOT_VERSION = "style_context_snapshot_v1"

_ASSIST_AXES = ("family", "weight", "scale")
_COMPATIBILITY_AXES = ("fill", "outline", "orientation")
_ALL_TRANSPORT_AXES = frozenset((*_ASSIST_AXES, *_COMPATIBILITY_AXES))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _thaw_json(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _identity(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


@dataclass(frozen=True)
class StyleContextPolicyIdentity:
    components: Mapping[str, Any]
    policy_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", _freeze_json(self.components))
        object.__setattr__(self, "policy_id", _required_text(self.policy_id, "policy_id"))

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "version": STYLE_CONTEXT_POLICY_VERSION,
            "policy_id": self.policy_id,
            "components": _thaw_json(self.components),
        }


@dataclass(frozen=True)
class StyleContextRunIdentity:
    import_root: str
    page_names: tuple[str, ...]
    source_language: str
    target_language: str
    run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "import_root",
            _required_text(self.import_root, "import_root"),
        )
        object.__setattr__(
            self,
            "page_names",
            tuple(_required_text(name, "page_name") for name in self.page_names),
        )
        object.__setattr__(
            self,
            "source_language",
            _required_text(self.source_language, "source_language"),
        )
        object.__setattr__(
            self,
            "target_language",
            _required_text(self.target_language, "target_language"),
        )
        object.__setattr__(self, "run_id", _required_text(self.run_id, "run_id"))

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "version": STYLE_CONTEXT_RUN_IDENTITY_VERSION,
            "run_id": self.run_id,
            "import_root": self.import_root,
            "page_names": list(self.page_names),
            "source_language": self.source_language,
            "target_language": self.target_language,
        }


@dataclass(frozen=True)
class StyleContextPageIdentity:
    page_index: int
    page_id: str
    page_name: str
    source_sha256: str

    def __post_init__(self) -> None:
        page_index = int(self.page_index)
        if page_index < 0:
            raise ValueError("page_index must be non-negative")
        object.__setattr__(self, "page_index", page_index)
        object.__setattr__(self, "page_id", _required_text(self.page_id, "page_id"))
        object.__setattr__(
            self,
            "page_name",
            _required_text(self.page_name, "page_name"),
        )
        object.__setattr__(
            self,
            "source_sha256",
            _required_text(self.source_sha256, "source_sha256"),
        )

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "version": STYLE_CONTEXT_PAGE_IDENTITY_VERSION,
            "page_index": self.page_index,
            "page_id": self.page_id,
            "page_name": self.page_name,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class StyleContextAxisRecord:
    axis: str
    value: Mapping[str, Any]
    confidence: float
    provenance: str
    support_identity_sha256: str

    def __post_init__(self) -> None:
        axis = str(self.axis or "").strip().lower()
        if axis not in _ALL_TRANSPORT_AXES:
            raise ValueError(f"unsupported style-context axis: {axis}")
        confidence = float(self.confidence)
        if not (0.0 < confidence <= 1.0):
            raise ValueError(f"invalid confidence for {axis}: {confidence}")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "value", _freeze_json(self.value))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "provenance",
            _required_text(self.provenance, f"{axis}_provenance"),
        )
        object.__setattr__(
            self,
            "support_identity_sha256",
            _required_text(
                self.support_identity_sha256,
                f"{axis}_support_identity_sha256",
            ),
        )

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "value": _thaw_json(self.value),
            "confidence": round(self.confidence, 8),
            "provenance": self.provenance,
            "support_identity_sha256": self.support_identity_sha256,
        }


@dataclass(frozen=True)
class StyleContextRecord:
    page_id: str
    bundle_id: str
    parent_id: str
    root_id: str
    role: str
    semantic_class: str
    target_font_affinity: Mapping[str, Any] = field(default_factory=dict)
    family_posterior: Mapping[str, Any] = field(default_factory=dict)
    assist_axes: tuple[StyleContextAxisRecord, ...] = ()
    compatibility_axes: tuple[StyleContextAxisRecord, ...] = ()
    evidence_identity_sha256: str = ""

    def __post_init__(self) -> None:
        for field_name in ("page_id", "bundle_id", "parent_id", "root_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "role", str(self.role or "").strip())
        object.__setattr__(
            self,
            "semantic_class",
            str(self.semantic_class or "").strip(),
        )
        object.__setattr__(
            self,
            "target_font_affinity",
            _freeze_json(self.target_font_affinity),
        )
        object.__setattr__(
            self,
            "family_posterior",
            _freeze_json(self.family_posterior),
        )
        object.__setattr__(self, "assist_axes", tuple(self.assist_axes or ()))
        object.__setattr__(
            self,
            "compatibility_axes",
            tuple(self.compatibility_axes or ()),
        )
        object.__setattr__(
            self,
            "evidence_identity_sha256",
            _required_text(
                self.evidence_identity_sha256,
                "evidence_identity_sha256",
            ),
        )

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "bundle_id": self.bundle_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "role": self.role,
            "semantic_class": self.semantic_class,
            "target_font_affinity": _thaw_json(self.target_font_affinity),
            "family_posterior": _thaw_json(self.family_posterior),
            "assist_axes": [item.to_project_dict() for item in self.assist_axes],
            "compatibility_axes": [
                item.to_project_dict() for item in self.compatibility_axes
            ],
            "evidence_identity_sha256": self.evidence_identity_sha256,
        }


@dataclass(frozen=True)
class StyleContextSnapshot:
    run_identity: StyleContextRunIdentity
    policy_identity: StyleContextPolicyIdentity
    page_index: int
    prefix_page_ids: tuple[str, ...]
    records: tuple[StyleContextRecord, ...]
    committed_delta_ids: tuple[str, ...]
    snapshot_id: str

    def __post_init__(self) -> None:
        page_index = int(self.page_index)
        if page_index < 0:
            raise ValueError("snapshot page_index must be non-negative")
        object.__setattr__(self, "page_index", page_index)
        object.__setattr__(
            self,
            "prefix_page_ids",
            tuple(str(item) for item in self.prefix_page_ids),
        )
        object.__setattr__(self, "records", tuple(self.records or ()))
        object.__setattr__(
            self,
            "committed_delta_ids",
            tuple(str(item) for item in self.committed_delta_ids),
        )
        object.__setattr__(
            self,
            "snapshot_id",
            _required_text(self.snapshot_id, "snapshot_id"),
        )

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "version": STYLE_CONTEXT_SNAPSHOT_VERSION,
            "run_id": self.run_identity.run_id,
            "policy_id": self.policy_identity.policy_id,
            "page_index": self.page_index,
            "prefix_page_ids": list(self.prefix_page_ids),
            "committed_delta_ids": list(self.committed_delta_ids),
            "snapshot_id": self.snapshot_id,
            "records": [record.to_project_dict() for record in self.records],
        }


@dataclass(frozen=True)
class StyleContextDelta:
    run_id: str
    policy_id: str
    base_snapshot_id: str
    page_identity: StyleContextPageIdentity
    records: tuple[StyleContextRecord, ...]
    delta_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _required_text(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "policy_id",
            _required_text(self.policy_id, "policy_id"),
        )
        object.__setattr__(
            self,
            "base_snapshot_id",
            _required_text(self.base_snapshot_id, "base_snapshot_id"),
        )
        object.__setattr__(self, "records", tuple(self.records or ()))
        object.__setattr__(
            self,
            "delta_id",
            _required_text(self.delta_id, "delta_id"),
        )

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "version": STYLE_CONTEXT_DELTA_VERSION,
            "run_id": self.run_id,
            "policy_id": self.policy_id,
            "base_snapshot_id": self.base_snapshot_id,
            "page_identity": self.page_identity.to_project_dict(),
            "records": [record.to_project_dict() for record in self.records],
            "delta_id": self.delta_id,
        }


@dataclass(frozen=True)
class StyleContextJournal:
    run_identity: StyleContextRunIdentity
    policy_identity: StyleContextPolicyIdentity
    committed_deltas: tuple[StyleContextDelta, ...] = ()
    journal_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "committed_deltas",
            tuple(self.committed_deltas or ()),
        )
        object.__setattr__(
            self,
            "journal_id",
            _required_text(self.journal_id, "journal_id"),
        )

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "version": STYLE_CONTEXT_CACHE_VERSION,
            "run_identity": self.run_identity.to_project_dict(),
            "policy_identity": self.policy_identity.to_project_dict(),
            "committed_deltas": [
                delta.to_project_dict() for delta in self.committed_deltas
            ],
            "journal_id": self.journal_id,
        }


@dataclass(frozen=True)
class StyleContextLoadResult:
    status: str
    journal: StyleContextJournal
    invalidation_reason: str = ""
    source_prefix_length: int = 0


def build_style_context_policy_identity(
    components: Mapping[str, Any],
) -> StyleContextPolicyIdentity:
    frozen = _freeze_json(dict(components or {}))
    payload = {
        "version": STYLE_CONTEXT_POLICY_VERSION,
        "components": _thaw_json(frozen),
    }
    return StyleContextPolicyIdentity(
        components=frozen,
        policy_id=_identity("style-policy", payload),
    )


def build_style_context_run_identity(
    *,
    import_dir: str,
    page_names: Iterable[str],
    source_language: str,
    target_language: str,
) -> StyleContextRunIdentity:
    normalized_root = ntpath.normcase(
        ntpath.abspath(str(import_dir or "").replace("/", "\\"))
    )
    names = tuple(str(name or "") for name in page_names)
    payload = {
        "version": STYLE_CONTEXT_RUN_IDENTITY_VERSION,
        "import_root": normalized_root,
        "page_names": list(names),
        "source_language": str(source_language or ""),
        "target_language": str(target_language or ""),
    }
    return StyleContextRunIdentity(
        import_root=normalized_root,
        page_names=names,
        source_language=str(source_language or ""),
        target_language=str(target_language or ""),
        run_id=_identity("style-run", payload),
    )


def _journal_id(
    run_identity: StyleContextRunIdentity,
    policy_identity: StyleContextPolicyIdentity,
    committed_deltas: Iterable[StyleContextDelta],
) -> str:
    return _identity(
        "style-journal",
        {
            "version": STYLE_CONTEXT_CACHE_VERSION,
            "run_id": run_identity.run_id,
            "policy_id": policy_identity.policy_id,
            "committed_delta_ids": [
                item.delta_id for item in tuple(committed_deltas)
            ],
        },
    )


def empty_style_context_journal(
    *,
    run_identity: StyleContextRunIdentity,
    policy_identity: StyleContextPolicyIdentity,
) -> StyleContextJournal:
    return StyleContextJournal(
        run_identity=run_identity,
        policy_identity=policy_identity,
        committed_deltas=(),
        journal_id=_journal_id(run_identity, policy_identity, ()),
    )


def style_context_snapshot_before(
    journal: StyleContextJournal,
    *,
    page_index: int,
) -> StyleContextSnapshot:
    requested_index = int(page_index)
    if requested_index < 0:
        raise ValueError("page_index must be non-negative")
    eligible = tuple(
        delta
        for delta in journal.committed_deltas
        if delta.page_identity.page_index < requested_index
    )
    prefix_page_ids = tuple(delta.page_identity.page_id for delta in eligible)
    records = tuple(record for delta in eligible for record in delta.records)
    committed_delta_ids = tuple(delta.delta_id for delta in eligible)
    snapshot_payload = {
        "version": STYLE_CONTEXT_SNAPSHOT_VERSION,
        "run_id": journal.run_identity.run_id,
        "policy_id": journal.policy_identity.policy_id,
        "page_index": requested_index,
        "committed_delta_ids": list(committed_delta_ids),
    }
    return StyleContextSnapshot(
        run_identity=journal.run_identity,
        policy_identity=journal.policy_identity,
        page_index=requested_index,
        prefix_page_ids=prefix_page_ids,
        records=records,
        committed_delta_ids=committed_delta_ids,
        snapshot_id=_identity("style-snapshot", snapshot_payload),
    )


def _axis_record_from_observation(
    observation: Any,
) -> StyleContextAxisRecord | None:
    axis = str(getattr(observation, "axis", "") or "").strip().lower()
    if axis not in _ALL_TRANSPORT_AXES:
        return None
    status = str(getattr(observation, "status", "") or "").strip().lower()
    confidence = float(getattr(observation, "confidence", 0.0) or 0.0)
    supported = bool(
        getattr(observation, "supported", status == "supported" and confidence > 0.0)
    )
    provenance = str(getattr(observation, "provenance", "") or "").strip()
    value = getattr(observation, "value", {})
    support_identity = getattr(observation, "support_identity", {})
    if (
        not supported
        or status != "supported"
        or confidence <= 0.0
        or not provenance
        or not isinstance(value, Mapping)
        or not isinstance(support_identity, Mapping)
        or not support_identity
    ):
        return None
    return StyleContextAxisRecord(
        axis=axis,
        value=value,
        confidence=confidence,
        provenance=provenance,
        support_identity_sha256=_identity(
            "style-support",
            support_identity,
        ),
    )


def _record_from_bundle_and_evidence(
    bundle: Any,
    evidence: Any,
) -> StyleContextRecord | None:
    if (
        str(getattr(evidence, "status", "") or "").strip().lower() != "observed"
        or not bool(getattr(evidence, "vote_eligible", False))
    ):
        return None
    identity_fields = ("page_id", "bundle_id", "parent_id", "root_id")
    bundle_identity = tuple(
        str(getattr(bundle, field_name, "") or "") for field_name in identity_fields
    )
    evidence_identity = tuple(
        str(getattr(evidence, field_name, "") or "") for field_name in identity_fields
    )
    if (
        not all(bundle_identity)
        or bundle_identity != evidence_identity
        or not bool(getattr(bundle, "render_required", False))
    ):
        return None

    qualified_by_axis: dict[str, StyleContextAxisRecord] = {}
    for item in tuple(getattr(evidence, "axis_evidence", ()) or ()):
        qualified = _axis_record_from_observation(item)
        if qualified is not None:
            qualified_by_axis[qualified.axis] = qualified
    assist_axes = tuple(
        qualified_by_axis[axis] for axis in _ASSIST_AXES if axis in qualified_by_axis
    )
    compatibility_axes = tuple(
        qualified_by_axis[axis]
        for axis in _COMPATIBILITY_AXES
        if axis in qualified_by_axis
    )
    target_font_affinity: Mapping[str, Any] = {}
    source_font_observation = getattr(
        evidence,
        "source_font_observation",
        None,
    )
    affinity = getattr(
        source_font_observation,
        "target_font_affinity",
        None,
    )
    affinity_to_audit = getattr(affinity, "to_audit_dict", None)
    if callable(affinity_to_audit):
        candidate = affinity_to_audit()
        detector_input_sha256 = str(
            getattr(evidence, "detector_input_sha256", "") or ""
        )
        if (
            isinstance(candidate, Mapping)
            and detector_input_sha256
            and str(candidate.get("source_input_sha256") or "")
            == detector_input_sha256
        ):
            target_font_affinity = candidate

    family_posterior: Mapping[str, Any] = {}
    posterior = getattr(evidence, "family_posterior", None)
    posterior_to_audit = getattr(posterior, "to_audit_dict", None)
    if callable(posterior_to_audit):
        candidate = posterior_to_audit()
        if isinstance(candidate, Mapping):
            family_posterior = candidate

    if (
        not assist_axes
        and not compatibility_axes
        and not target_font_affinity
        and not family_posterior
    ):
        return None
    record_payload = {
        "identity": list(bundle_identity),
        "target_font_affinity": dict(target_font_affinity),
        "family_posterior": dict(family_posterior),
        "assist": [item.to_project_dict() for item in assist_axes],
        "compatibility": [
            item.to_project_dict() for item in compatibility_axes
        ],
    }
    return StyleContextRecord(
        page_id=bundle_identity[0],
        bundle_id=bundle_identity[1],
        parent_id=bundle_identity[2],
        root_id=bundle_identity[3],
        role=str(getattr(bundle, "role", "") or ""),
        semantic_class=str(getattr(bundle, "semantic_class", "") or ""),
        target_font_affinity=target_font_affinity,
        family_posterior=family_posterior,
        assist_axes=assist_axes,
        compatibility_axes=compatibility_axes,
        evidence_identity_sha256=_identity("style-evidence", record_payload),
    )


def _delta_id(
    *,
    run_id: str,
    policy_id: str,
    base_snapshot_id: str,
    page_identity: StyleContextPageIdentity,
    records: Iterable[StyleContextRecord],
) -> str:
    return _identity(
        "style-delta",
        {
            "version": STYLE_CONTEXT_DELTA_VERSION,
            "run_id": run_id,
            "policy_id": policy_id,
            "base_snapshot_id": base_snapshot_id,
            "page_identity": page_identity.to_project_dict(),
            "records": [record.to_project_dict() for record in tuple(records)],
        },
    )


def prepare_style_context_delta(
    *,
    snapshot: StyleContextSnapshot,
    page_identity: StyleContextPageIdentity,
    parent_execution_bundles: Iterable[Any],
    evidence: Iterable[Any],
) -> StyleContextDelta:
    if page_identity.page_index != snapshot.page_index:
        raise ValueError("page identity does not match requested prefix snapshot")
    bundles_by_id: dict[str, Any] = {}
    for bundle in tuple(parent_execution_bundles or ()):
        if not bool(getattr(bundle, "render_required", False)):
            continue
        bundle_id = _required_text(
            getattr(bundle, "bundle_id", ""),
            "bundle_id",
        )
        if bundle_id in bundles_by_id:
            raise ValueError(f"duplicate bundle identity: {bundle_id}")
        bundles_by_id[bundle_id] = bundle
    evidence_by_id: dict[str, Any] = {}
    for item in tuple(evidence or ()):
        bundle_id = str(getattr(item, "bundle_id", "") or "").strip()
        if not bundle_id:
            continue
        if bundle_id in evidence_by_id:
            raise ValueError(f"duplicate evidence identity: {bundle_id}")
        evidence_by_id[bundle_id] = item

    records: list[StyleContextRecord] = []
    for bundle_id in sorted(bundles_by_id):
        item = evidence_by_id.get(bundle_id)
        if item is None:
            continue
        record = _record_from_bundle_and_evidence(bundles_by_id[bundle_id], item)
        if record is not None and record.page_id == page_identity.page_id:
            records.append(record)
    frozen_records = tuple(records)
    delta_id = _delta_id(
        run_id=snapshot.run_identity.run_id,
        policy_id=snapshot.policy_identity.policy_id,
        base_snapshot_id=snapshot.snapshot_id,
        page_identity=page_identity,
        records=frozen_records,
    )
    return StyleContextDelta(
        run_id=snapshot.run_identity.run_id,
        policy_id=snapshot.policy_identity.policy_id,
        base_snapshot_id=snapshot.snapshot_id,
        page_identity=page_identity,
        records=frozen_records,
        delta_id=delta_id,
    )


def journal_with_committed_delta(
    journal: StyleContextJournal,
    delta: StyleContextDelta,
) -> StyleContextJournal:
    if delta.run_id != journal.run_identity.run_id:
        raise ValueError("delta run identity mismatch")
    if delta.policy_id != journal.policy_identity.policy_id:
        raise ValueError("delta policy identity mismatch")
    page_index = delta.page_identity.page_index
    if page_index > len(journal.committed_deltas):
        raise ValueError("delta would create a cache-prefix gap")
    expected_snapshot = style_context_snapshot_before(
        journal,
        page_index=page_index,
    )
    if delta.base_snapshot_id != expected_snapshot.snapshot_id:
        raise ValueError("delta base snapshot mismatch")
    candidate = (*journal.committed_deltas[:page_index], delta)
    return StyleContextJournal(
        run_identity=journal.run_identity,
        policy_identity=journal.policy_identity,
        committed_deltas=candidate,
        journal_id=_journal_id(
            journal.run_identity,
            journal.policy_identity,
            candidate,
        ),
    )


def _axis_record_from_project(value: Mapping[str, Any]) -> StyleContextAxisRecord:
    return StyleContextAxisRecord(
        axis=value.get("axis", ""),
        value=value.get("value", {}),
        confidence=value.get("confidence", 0.0),
        provenance=value.get("provenance", ""),
        support_identity_sha256=value.get("support_identity_sha256", ""),
    )


def _record_from_project(value: Mapping[str, Any]) -> StyleContextRecord:
    return StyleContextRecord(
        page_id=value.get("page_id", ""),
        bundle_id=value.get("bundle_id", ""),
        parent_id=value.get("parent_id", ""),
        root_id=value.get("root_id", ""),
        role=value.get("role", ""),
        semantic_class=value.get("semantic_class", ""),
        target_font_affinity=(
            value.get("target_font_affinity", {})
            if isinstance(value.get("target_font_affinity"), Mapping)
            else {}
        ),
        family_posterior=(
            value.get("family_posterior", {})
            if isinstance(value.get("family_posterior"), Mapping)
            else {}
        ),
        assist_axes=tuple(
            _axis_record_from_project(item)
            for item in value.get("assist_axes", ())
            if isinstance(item, Mapping)
        ),
        compatibility_axes=tuple(
            _axis_record_from_project(item)
            for item in value.get("compatibility_axes", ())
            if isinstance(item, Mapping)
        ),
        evidence_identity_sha256=value.get("evidence_identity_sha256", ""),
    )


def _page_identity_from_project(
    value: Mapping[str, Any],
) -> StyleContextPageIdentity:
    if value.get("version") != STYLE_CONTEXT_PAGE_IDENTITY_VERSION:
        raise ValueError("page identity version mismatch")
    return StyleContextPageIdentity(
        page_index=value.get("page_index", -1),
        page_id=value.get("page_id", ""),
        page_name=value.get("page_name", ""),
        source_sha256=value.get("source_sha256", ""),
    )


def _delta_from_project(
    value: Mapping[str, Any],
    *,
    snapshot: StyleContextSnapshot,
) -> StyleContextDelta:
    if value.get("version") != STYLE_CONTEXT_DELTA_VERSION:
        raise ValueError("delta version mismatch")
    page_identity_value = value.get("page_identity")
    if not isinstance(page_identity_value, Mapping):
        raise ValueError("delta page identity is missing")
    page_identity = _page_identity_from_project(page_identity_value)
    records = tuple(
        _record_from_project(item)
        for item in value.get("records", ())
        if isinstance(item, Mapping)
    )
    delta = StyleContextDelta(
        run_id=value.get("run_id", ""),
        policy_id=value.get("policy_id", ""),
        base_snapshot_id=value.get("base_snapshot_id", ""),
        page_identity=page_identity,
        records=records,
        delta_id=value.get("delta_id", ""),
    )
    expected_id = _delta_id(
        run_id=delta.run_id,
        policy_id=delta.policy_id,
        base_snapshot_id=delta.base_snapshot_id,
        page_identity=delta.page_identity,
        records=delta.records,
    )
    if delta.delta_id != expected_id:
        raise ValueError("delta digest mismatch")
    if delta.base_snapshot_id != snapshot.snapshot_id:
        raise ValueError("delta prefix identity mismatch")
    return delta


def _invalidated(
    *,
    run_identity: StyleContextRunIdentity,
    policy_identity: StyleContextPolicyIdentity,
    reason: str,
    source_prefix_length: int = 0,
) -> StyleContextLoadResult:
    return StyleContextLoadResult(
        status="invalidated",
        journal=empty_style_context_journal(
            run_identity=run_identity,
            policy_identity=policy_identity,
        ),
        invalidation_reason=reason,
        source_prefix_length=source_prefix_length,
    )


def load_style_context_journal(
    payload: Any,
    *,
    run_identity: StyleContextRunIdentity,
    policy_identity: StyleContextPolicyIdentity,
) -> StyleContextLoadResult:
    if payload is None:
        return StyleContextLoadResult(
            status="empty",
            journal=empty_style_context_journal(
                run_identity=run_identity,
                policy_identity=policy_identity,
            ),
            invalidation_reason="cache_missing",
        )
    if not isinstance(payload, Mapping):
        return _invalidated(
            run_identity=run_identity,
            policy_identity=policy_identity,
            reason="malformed_cache",
        )
    raw_deltas = payload.get("committed_deltas", ())
    source_prefix_length = len(raw_deltas) if isinstance(raw_deltas, list) else 0
    if payload.get("version") != STYLE_CONTEXT_CACHE_VERSION:
        return _invalidated(
            run_identity=run_identity,
            policy_identity=policy_identity,
            reason="cache_version_mismatch",
            source_prefix_length=source_prefix_length,
        )
    raw_policy = payload.get("policy_identity")
    if (
        not isinstance(raw_policy, Mapping)
        or raw_policy.get("policy_id") != policy_identity.policy_id
    ):
        return _invalidated(
            run_identity=run_identity,
            policy_identity=policy_identity,
            reason="policy_identity_mismatch",
            source_prefix_length=source_prefix_length,
        )
    raw_run = payload.get("run_identity")
    if (
        not isinstance(raw_run, Mapping)
        or raw_run.get("run_id") != run_identity.run_id
    ):
        return _invalidated(
            run_identity=run_identity,
            policy_identity=policy_identity,
            reason="run_identity_mismatch",
            source_prefix_length=source_prefix_length,
        )
    if not isinstance(raw_deltas, list):
        return _invalidated(
            run_identity=run_identity,
            policy_identity=policy_identity,
            reason="malformed_cache",
        )

    try:
        journal = empty_style_context_journal(
            run_identity=run_identity,
            policy_identity=policy_identity,
        )
        for expected_index, item in enumerate(raw_deltas):
            if not isinstance(item, Mapping):
                raise ValueError("delta is not a mapping")
            snapshot = style_context_snapshot_before(
                journal,
                page_index=expected_index,
            )
            delta = _delta_from_project(item, snapshot=snapshot)
            if delta.page_identity.page_index != expected_index:
                raise ValueError("delta order mismatch")
            if expected_index >= len(run_identity.page_names):
                raise ValueError("delta exceeds selected run")
            if (
                delta.page_identity.page_name
                != run_identity.page_names[expected_index]
            ):
                raise ValueError("delta page name mismatch")
            journal = journal_with_committed_delta(journal, delta)
        if payload.get("journal_id") != journal.journal_id:
            raise ValueError("journal digest mismatch")
    except (TypeError, ValueError):
        return _invalidated(
            run_identity=run_identity,
            policy_identity=policy_identity,
            reason="malformed_cache",
            source_prefix_length=source_prefix_length,
        )
    return StyleContextLoadResult(
        status="loaded",
        journal=journal,
        source_prefix_length=len(journal.committed_deltas),
    )
