# -*- coding: utf-8 -*-
"""Immutable contracts for the Project Edit Ledger.

The contracts in this module describe user-authored intent only.  They do not
run pipeline owners, mutate automated records, or interpret renderer output.
"""
from __future__ import annotations

from dataclasses import InitVar, dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

from app.pipeline.hierarchy_revision_contracts import (
    EFFECTIVE_HIERARCHY_REVISION_PREFIX,
    ParentStageRequirement,
    USER_PARENT_ID_PREFIX,
    USER_PARENT_IDENTITY_NAMESPACE,
    USER_ROOT_ID_PREFIX,
    USER_ROOT_IDENTITY_NAMESPACE,
    validate_user_parent_identity_pair,
    user_root_identity_suffix,
)
from app.pipeline.ocr_revision_contracts import (
    OCR_SOURCE_REVISION_ID_PREFIX,
    OCR_SOURCE_SELECTION_EDIT_ID_PREFIX,
)
from app.pipeline.translation_revision_contracts import (
    TRANSLATION_REVISION_ID_PREFIX,
    TRANSLATION_SELECTION_EDIT_ID_PREFIX,
)
from app.render.parent_layer_effects import (
    ROTATION_MAX_DEGREES,
    ROTATION_MIN_DEGREES,
    SHADOW_BLUR_MAX_PX,
    SHADOW_OFFSET_LIMIT_PX,
)
from app.render.font_manager import REQUIRED_FONT_ROLES


EDIT_SCHEMA_VERSION = "project_edit_v1"
LEDGER_SCHEMA_VERSION = "project_edit_ledger_v1"
SOURCE_TEXT_REVISION_BASE_SCHEMA_VERSION = "source_text_revision_base_v1"
TARGET_TEXT_REVISION_BASE_SCHEMA_VERSION = "target_text_revision_base_v1"
PARENT_SOURCE_EVIDENCE_MAPPING_SCHEMA_VERSION = (
    "parent_source_evidence_mapping_v1"
)


class EditTargetKind(str, Enum):
    PROJECT = "project"
    PAGE = "page"
    PARENT = "parent"
    ARTIFACT = "artifact"
    EDIT = "edit"


class EditDomain(str, Enum):
    STRUCTURAL = "structural"
    SOURCE_TEXT = "source_text"
    TARGET_TEXT = "target_text"
    CLEANUP = "cleanup"
    RENDER_STYLE = "render_style"
    RENDER_LAYOUT = "render_layout"
    REVIEW_METADATA = "review_metadata"
    GLOSSARY = "glossary"
    LEDGER_CONTROL = "_ledger_control"


@dataclass(frozen=True, slots=True)
class SourceTextRevisionBaseV1:
    """Immutable selected-model base for one user-parent source override."""

    source_revision_id: str
    selection_edit_id: str
    artifact_sha256: str
    source_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    schema_version: str = SOURCE_TEXT_REVISION_BASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_TEXT_REVISION_BASE_SCHEMA_VERSION:
            raise ValueError("unsupported source-text revision-base schema")
        prefixes = {
            "source_revision_id": OCR_SOURCE_REVISION_ID_PREFIX,
            "selection_edit_id": OCR_SOURCE_SELECTION_EDIT_ID_PREFIX,
            "hierarchy_revision_id": EFFECTIVE_HIERARCHY_REVISION_PREFIX,
        }
        for field_name, prefix in prefixes.items():
            value = _require_non_empty(getattr(self, field_name), field_name)
            if not value.startswith(prefix):
                raise ValueError(f"{field_name} is invalid")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "artifact_sha256",
            "source_fingerprint",
            "hierarchy_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "source_revision_id": self.source_revision_id,
            "selection_edit_id": self.selection_edit_id,
            "artifact_sha256": self.artifact_sha256,
            "source_fingerprint": self.source_fingerprint,
            "hierarchy_revision_id": self.hierarchy_revision_id,
            "hierarchy_fingerprint": self.hierarchy_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceTextRevisionBaseV1":
        if not isinstance(value, Mapping):
            raise TypeError("source-text revision base must be a mapping")
        expected = {
            "schema_version",
            "source_revision_id",
            "selection_edit_id",
            "artifact_sha256",
            "source_fingerprint",
            "hierarchy_revision_id",
            "hierarchy_fingerprint",
        }
        if set(value) != expected:
            raise ValueError("source-text revision-base fields are invalid")
        return cls(
            schema_version=str(value.get("schema_version") or ""),
            source_revision_id=str(value.get("source_revision_id") or ""),
            selection_edit_id=str(value.get("selection_edit_id") or ""),
            artifact_sha256=str(value.get("artifact_sha256") or ""),
            source_fingerprint=str(value.get("source_fingerprint") or ""),
            hierarchy_revision_id=str(
                value.get("hierarchy_revision_id") or ""
            ),
            hierarchy_fingerprint=str(
                value.get("hierarchy_fingerprint") or ""
            ),
        )


@dataclass(frozen=True, slots=True)
class TargetTextRevisionBaseV1:
    """Immutable selected-model base for one user-parent target override."""

    translation_revision_id: str
    selection_edit_id: str
    artifact_sha256: str
    source_fingerprint: str
    source_revision_id: str
    source_selection_edit_id: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    schema_version: str = TARGET_TEXT_REVISION_BASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_TEXT_REVISION_BASE_SCHEMA_VERSION:
            raise ValueError("unsupported target-text revision-base schema")
        prefixes = {
            "translation_revision_id": TRANSLATION_REVISION_ID_PREFIX,
            "selection_edit_id": TRANSLATION_SELECTION_EDIT_ID_PREFIX,
            "source_revision_id": OCR_SOURCE_REVISION_ID_PREFIX,
            "source_selection_edit_id": OCR_SOURCE_SELECTION_EDIT_ID_PREFIX,
            "hierarchy_revision_id": EFFECTIVE_HIERARCHY_REVISION_PREFIX,
        }
        for field_name, prefix in prefixes.items():
            value = _require_non_empty(getattr(self, field_name), field_name)
            if not value.startswith(prefix):
                raise ValueError(f"{field_name} is invalid")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "artifact_sha256",
            "source_fingerprint",
            "hierarchy_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "translation_revision_id": self.translation_revision_id,
            "selection_edit_id": self.selection_edit_id,
            "artifact_sha256": self.artifact_sha256,
            "source_fingerprint": self.source_fingerprint,
            "source_revision_id": self.source_revision_id,
            "source_selection_edit_id": self.source_selection_edit_id,
            "hierarchy_revision_id": self.hierarchy_revision_id,
            "hierarchy_fingerprint": self.hierarchy_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetTextRevisionBaseV1":
        if not isinstance(value, Mapping):
            raise TypeError("target-text revision base must be a mapping")
        expected = {
            "schema_version",
            "translation_revision_id",
            "selection_edit_id",
            "artifact_sha256",
            "source_fingerprint",
            "source_revision_id",
            "source_selection_edit_id",
            "hierarchy_revision_id",
            "hierarchy_fingerprint",
        }
        if set(value) != expected:
            raise ValueError("target-text revision-base fields are invalid")
        return cls(
            schema_version=str(value.get("schema_version") or ""),
            translation_revision_id=str(
                value.get("translation_revision_id") or ""
            ),
            selection_edit_id=str(value.get("selection_edit_id") or ""),
            artifact_sha256=str(value.get("artifact_sha256") or ""),
            source_fingerprint=str(value.get("source_fingerprint") or ""),
            source_revision_id=str(value.get("source_revision_id") or ""),
            source_selection_edit_id=str(
                value.get("source_selection_edit_id") or ""
            ),
            hierarchy_revision_id=str(
                value.get("hierarchy_revision_id") or ""
            ),
            hierarchy_fingerprint=str(
                value.get("hierarchy_fingerprint") or ""
            ),
        )


@dataclass(frozen=True, slots=True)
class ParentSourceEvidenceMappingV1:
    """Application-owned mapping to immutable detected-parent OCR evidence.

    The mapping does not create or copy a pipeline bundle.  It records exactly
    which existing automatic parents supply identity, geometry, OCR text,
    reading order, and reusable render evidence for one effective user parent.
    """

    page_id: str
    source_parent_ids: tuple[str, ...]
    source_root_ids: tuple[str, ...]
    source_bundle_ids: tuple[str, ...]
    source_automatic_fingerprints: tuple[str, ...]
    source_bboxes: tuple[tuple[int, int, int, int], ...]
    source_texts: tuple[str, ...]
    source_text_fingerprints: tuple[str, ...]
    source_target_texts: tuple[str, ...]
    source_target_text_fingerprints: tuple[str, ...]
    source_reading_orders: tuple[int, ...]
    source_roles: tuple[str, ...]
    primary_source_parent_id: str
    schema_version: str = PARENT_SOURCE_EVIDENCE_MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PARENT_SOURCE_EVIDENCE_MAPPING_SCHEMA_VERSION:
            raise ValueError("unsupported parent source-evidence mapping schema")
        object.__setattr__(
            self,
            "page_id",
            _require_non_empty(self.page_id, "page_id"),
        )
        field_lengths = {
            len(self.source_parent_ids),
            len(self.source_root_ids),
            len(self.source_bundle_ids),
            len(self.source_automatic_fingerprints),
            len(self.source_bboxes),
            len(self.source_texts),
            len(self.source_text_fingerprints),
            len(self.source_target_texts),
            len(self.source_target_text_fingerprints),
            len(self.source_reading_orders),
            len(self.source_roles),
        }
        if len(field_lengths) != 1 or not field_lengths or next(iter(field_lengths)) < 1:
            raise ValueError(
                "parent source-evidence mapping fields must have one shared non-zero length"
            )
        normalized_parent_ids = tuple(
            _require_non_empty(value, f"source_parent_ids[{index}]")
            for index, value in enumerate(self.source_parent_ids)
        )
        if len(set(normalized_parent_ids)) != len(normalized_parent_ids):
            raise ValueError("source_parent_ids must be unique")
        object.__setattr__(self, "source_parent_ids", normalized_parent_ids)
        for field_name in ("source_root_ids", "source_bundle_ids"):
            values = tuple(
                _require_non_empty(value, f"{field_name}[{index}]")
                for index, value in enumerate(getattr(self, field_name))
            )
            if field_name == "source_bundle_ids" and len(set(values)) != len(values):
                raise ValueError(f"{field_name} must be unique")
            object.__setattr__(self, field_name, values)
        for field_name in (
            "source_automatic_fingerprints",
            "source_text_fingerprints",
            "source_target_text_fingerprints",
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(
                    _require_sha256(value, f"{field_name}[{index}]")
                    for index, value in enumerate(getattr(self, field_name))
                ),
            )
        normalized_bboxes: list[tuple[int, int, int, int]] = []
        for index, value in enumerate(self.source_bboxes):
            if (
                not isinstance(value, tuple)
                or len(value) != 4
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                )
            ):
                raise ValueError(
                    f"source_bboxes[{index}] must contain exact integer x, y, width, height"
                )
            bbox = tuple(int(item) for item in value)
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"source_bboxes[{index}] is invalid")
            normalized_bboxes.append(bbox)  # type: ignore[arg-type]
        object.__setattr__(self, "source_bboxes", tuple(normalized_bboxes))
        normalized_texts = tuple(
            str(value) if isinstance(value, str) else ""
            for value in self.source_texts
        )
        if any(not value or not value.strip() for value in normalized_texts):
            raise ValueError("source_texts must contain exact non-empty OCR text")
        object.__setattr__(self, "source_texts", normalized_texts)
        if any(not isinstance(value, str) for value in self.source_target_texts):
            raise TypeError("source_target_texts must contain exact pipeline text")
        normalized_target_texts = tuple(self.source_target_texts)
        object.__setattr__(self, "source_target_texts", normalized_target_texts)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in self.source_reading_orders
        ):
            raise ValueError(
                "source_reading_orders must contain non-negative exact integers"
            )
        normalized_orders = tuple(int(value) for value in self.source_reading_orders)
        if (
            len(set(normalized_orders)) != len(normalized_orders)
            or normalized_orders != tuple(sorted(normalized_orders))
        ):
            raise ValueError(
                "source_reading_orders must be unique and canonically sorted"
            )
        object.__setattr__(self, "source_reading_orders", normalized_orders)
        normalized_roles = tuple(str(value) for value in self.source_roles)
        if (
            any(value not in {"speech", "caption"} for value in normalized_roles)
            or len(set(normalized_roles)) != 1
        ):
            raise ValueError(
                "source_roles must contain one compatible speech or caption role"
            )
        object.__setattr__(self, "source_roles", normalized_roles)
        primary = _require_non_empty(
            self.primary_source_parent_id,
            "primary_source_parent_id",
        )
        if primary not in normalized_parent_ids:
            raise ValueError("primary_source_parent_id must name a mapped source")
        object.__setattr__(self, "primary_source_parent_id", primary)

    @property
    def source_text(self) -> str:
        return "".join(self.source_texts)

    @property
    def target_text(self) -> str | None:
        if any(not value.strip() for value in self.source_target_texts):
            return None
        return "".join(self.source_target_texts)

    @property
    def workflow_bbox(self) -> tuple[int, int, int, int]:
        left = min(bbox[0] for bbox in self.source_bboxes)
        top = min(bbox[1] for bbox in self.source_bboxes)
        right = max(bbox[0] + bbox[2] for bbox in self.source_bboxes)
        bottom = max(bbox[1] + bbox[3] for bbox in self.source_bboxes)
        return (left, top, right - left, bottom - top)

    @property
    def fingerprint(self) -> str:
        from .fingerprints import canonical_sha256

        return canonical_sha256(self._body())

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "page_id": self.page_id,
            "source_parent_ids": list(self.source_parent_ids),
            "source_root_ids": list(self.source_root_ids),
            "source_bundle_ids": list(self.source_bundle_ids),
            "source_automatic_fingerprints": list(
                self.source_automatic_fingerprints
            ),
            "source_bboxes": [list(value) for value in self.source_bboxes],
            "source_texts": list(self.source_texts),
            "source_text_fingerprints": list(self.source_text_fingerprints),
            "source_target_texts": list(self.source_target_texts),
            "source_target_text_fingerprints": list(
                self.source_target_text_fingerprints
            ),
            "source_reading_orders": list(self.source_reading_orders),
            "source_roles": list(self.source_roles),
            "primary_source_parent_id": self.primary_source_parent_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "mapping_fingerprint": self.fingerprint}

    def partition(
        self,
        child_bboxes: tuple[
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ],
    ) -> tuple["ParentSourceEvidenceMappingV1", "ParentSourceEvidenceMappingV1"]:
        """Partition mapped detection/OCR evidence without inventing a source."""

        if len(child_bboxes) != 2:
            raise ValueError("source evidence must be partitioned into two children")
        child_indices: list[list[int]] = [[], []]
        for source_index, source_bbox in enumerate(self.source_bboxes):
            matches = tuple(
                child_index
                for child_index, child_bbox in enumerate(child_bboxes)
                if (
                    source_bbox[0] >= child_bbox[0]
                    and source_bbox[1] >= child_bbox[1]
                    and source_bbox[0] + source_bbox[2]
                    <= child_bbox[0] + child_bbox[2]
                    and source_bbox[1] + source_bbox[3]
                    <= child_bbox[1] + child_bbox[3]
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    "each mapped detection source must be wholly contained in exactly one child"
                )
            child_indices[matches[0]].append(source_index)
        if any(not indices for indices in child_indices):
            raise ValueError("each split child must retain mapped detection/OCR evidence")

        def subset(indices: list[int]) -> ParentSourceEvidenceMappingV1:
            return ParentSourceEvidenceMappingV1(
                page_id=self.page_id,
                source_parent_ids=tuple(self.source_parent_ids[index] for index in indices),
                source_root_ids=tuple(self.source_root_ids[index] for index in indices),
                source_bundle_ids=tuple(self.source_bundle_ids[index] for index in indices),
                source_automatic_fingerprints=tuple(
                    self.source_automatic_fingerprints[index] for index in indices
                ),
                source_bboxes=tuple(self.source_bboxes[index] for index in indices),
                source_texts=tuple(self.source_texts[index] for index in indices),
                source_text_fingerprints=tuple(
                    self.source_text_fingerprints[index] for index in indices
                ),
                source_target_texts=tuple(
                    self.source_target_texts[index] for index in indices
                ),
                source_target_text_fingerprints=tuple(
                    self.source_target_text_fingerprints[index] for index in indices
                ),
                source_reading_orders=tuple(
                    self.source_reading_orders[index] for index in indices
                ),
                source_roles=tuple(self.source_roles[index] for index in indices),
                primary_source_parent_id=self.source_parent_ids[indices[0]],
            )

        return subset(child_indices[0]), subset(child_indices[1])

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParentSourceEvidenceMappingV1":
        if not isinstance(value, Mapping):
            raise TypeError("parent source-evidence mapping must be a mapping")
        expected = {
            "schema_version",
            "page_id",
            "source_parent_ids",
            "source_root_ids",
            "source_bundle_ids",
            "source_automatic_fingerprints",
            "source_bboxes",
            "source_texts",
            "source_text_fingerprints",
            "source_target_texts",
            "source_target_text_fingerprints",
            "source_reading_orders",
            "source_roles",
            "primary_source_parent_id",
            "mapping_fingerprint",
        }
        if set(value) != expected:
            raise ValueError("parent source-evidence mapping fields are invalid")
        result = cls(
            schema_version=str(value.get("schema_version") or ""),
            page_id=str(value.get("page_id") or ""),
            source_parent_ids=tuple(str(item) for item in value["source_parent_ids"]),
            source_root_ids=tuple(str(item) for item in value["source_root_ids"]),
            source_bundle_ids=tuple(str(item) for item in value["source_bundle_ids"]),
            source_automatic_fingerprints=tuple(
                str(item) for item in value["source_automatic_fingerprints"]
            ),
            source_bboxes=tuple(
                tuple(item for item in bbox)  # type: ignore[misc]
                for bbox in value["source_bboxes"]
            ),
            source_texts=tuple(str(item) for item in value["source_texts"]),
            source_text_fingerprints=tuple(
                str(item) for item in value["source_text_fingerprints"]
            ),
            source_target_texts=tuple(
                str(item) for item in value["source_target_texts"]
            ),
            source_target_text_fingerprints=tuple(
                str(item) for item in value["source_target_text_fingerprints"]
            ),
            source_reading_orders=tuple(
                item for item in value["source_reading_orders"]
            ),
            source_roles=tuple(str(item) for item in value["source_roles"]),
            primary_source_parent_id=str(
                value.get("primary_source_parent_id") or ""
            ),
        )
        if str(value.get("mapping_fingerprint") or "") != result.fingerprint:
            raise ValueError("parent source-evidence mapping fingerprint is stale")
        return result


_STYLE_FIELDS = frozenset(
    {
        "font_role",
        "font_weight_tier",
        "font_family",
        "font_face",
        "font_weight",
        "preferred_size",
        "minimum_size",
        "maximum_size",
        "fill_color",
        "outline_color",
        "outline_width",
        "shadow_enabled",
        "shadow_color",
        "shadow_offset",
        "shadow_blur",
        "shadow_spread",
        "shadow_opacity",
    }
)
_LAYOUT_FIELDS = frozenset(
    {
        "x",
        "y",
        "width",
        "height",
        "rotation",
        "writing_mode",
        "alignment",
        "line_height",
        "letter_spacing",
        "column_spacing",
        "run_spacing",
        "render_box",
        "break_hints",
    }
)

CANONICAL_WRITING_MODES = frozenset({"horizontal", "vertical"})
RENDER_LINE_HEIGHT_MIN = 0.5
RENDER_LINE_HEIGHT_MAX = 10.0
RENDER_ROTATION_MIN = ROTATION_MIN_DEGREES
RENDER_ROTATION_MAX = ROTATION_MAX_DEGREES
RENDER_OUTLINE_WIDTH_MIN = 0.0
RENDER_OUTLINE_WIDTH_MAX = 128.0
RENDER_PREFERRED_SIZE_MIN = 0.1
RENDER_PREFERRED_SIZE_MAX = 2048.0
RENDER_SHADOW_BLUR_MIN = 0.0
RENDER_SHADOW_BLUR_MAX = SHADOW_BLUR_MAX_PX
RENDER_SHADOW_OFFSET_MIN = -SHADOW_OFFSET_LIMIT_PX
RENDER_SHADOW_OFFSET_MAX = SHADOW_OFFSET_LIMIT_PX
REGISTERED_RENDER_FONT_ROLES = frozenset(
    str(item[0])
    for item in REQUIRED_FONT_ROLES
    if str(item[0]).startswith(("sans_", "serif_"))
)
RENDER_FONT_WEIGHT_TIERS = frozenset(
    {"slender", "base", "emphasis", "heavy"}
)


def _require_non_empty(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _require_sha256(value: Any, field_name: str) -> str:
    text = _require_non_empty(value, field_name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return text


def _validate_created_at(value: Any) -> str:
    text = _require_non_empty(value, "created_at")
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at must use UTC")
    return text


def _freeze_json(value: Any, *, field_name: str = "payload") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            frozen[key] = _freeze_json(item, field_name=f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(item, field_name=f"{field_name}[]") for item in value
        )
    raise TypeError(f"{field_name} is not JSON-compatible")


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def freeze_json(value: Any, *, field_name: str = "value") -> Any:
    """Expose the contract's immutable JSON representation to projections."""

    return _freeze_json(value, field_name=field_name)


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    keys = frozenset(payload)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"edit payload is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"edit payload has unsupported fields: {sorted(unknown)}")


def _require_fields_mapping(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str],
) -> None:
    _require_exact_keys(payload, required=frozenset({"fields"}))
    fields = payload.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("edit payload fields must be a non-empty mapping")
    if len(fields) != 1:
        raise ValueError(
            "render overrides must contain exactly one field per edit record"
        )
    unknown = frozenset(str(key) for key in fields) - allowed
    if unknown:
        raise ValueError(f"unsupported override fields: {sorted(unknown)}")


def _require_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}")
    return result


def canonical_render_line_height(
    value: Any,
    *,
    field_name: str = "line_height",
) -> float:
    """Return one exact renderer-supported line-height ratio."""

    return _require_number(
        value,
        field_name,
        minimum=RENDER_LINE_HEIGHT_MIN,
        maximum=RENDER_LINE_HEIGHT_MAX,
    )


def canonical_render_rotation(
    value: Any,
    *,
    field_name: str = "rotation",
) -> float:
    """Return one exact renderer-supported clockwise rotation in degrees."""

    return _require_number(
        value,
        field_name,
        minimum=RENDER_ROTATION_MIN,
        maximum=RENDER_ROTATION_MAX,
    )


def canonical_render_fill_color(
    value: Any,
    *,
    field_name: str = "fill_color",
) -> str:
    """Return one canonical opaque renderer fill color.

    This GUI-owned semantic deliberately excludes alpha.  The renderer's
    current glyph path cannot realize a caller-supplied color alpha exactly,
    so accepting ``#RRGGBBAA`` here would create false edit precision.
    """

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an exact hex color string")
    if len(value) != 7 or not value.startswith("#"):
        raise ValueError(f"{field_name} must use exactly #RRGGBB")
    if any(character not in "0123456789abcdefABCDEF" for character in value[1:]):
        raise ValueError(f"{field_name} must use hexadecimal digits")
    return value.upper()


def canonical_render_outline_color(
    value: Any,
    *,
    field_name: str = "outline_color",
) -> str:
    """Return one canonical opaque renderer outline color.

    The compositor uses glyph geometry as the final outline alpha mask, so a
    caller-supplied color alpha cannot be represented exactly.  New GUI-owned
    outline edits therefore share the strict opaque ``#RRGGBB`` contract.
    """

    return canonical_render_fill_color(value, field_name=field_name)


def canonical_render_shadow_color(
    value: Any,
    *,
    field_name: str = "shadow_color",
) -> str:
    """Return one canonical renderer-supported shadow RGBA color.

    The complete-parent shadow compositor realizes the color alpha exactly.
    Six-digit RGB is therefore canonical opaque RGBA rather than a separate
    semantic representation.
    """

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an exact hex color string")
    if len(value) not in {7, 9} or not value.startswith("#"):
        raise ValueError(f"{field_name} must use exactly #RRGGBB or #RRGGBBAA")
    if any(character not in "0123456789abcdefABCDEF" for character in value[1:]):
        raise ValueError(f"{field_name} must use hexadecimal digits")
    result = value.upper()
    return result if len(result) == 9 else f"{result}FF"


def canonical_render_outline_width(
    value: Any,
    *,
    field_name: str = "outline_width",
) -> float:
    """Return one exact renderer-supported outline stroke width in pixels."""

    result = _require_number(value, field_name)
    if not RENDER_OUTLINE_WIDTH_MIN <= result <= RENDER_OUTLINE_WIDTH_MAX:
        raise ValueError(
            f"{field_name} must be between {RENDER_OUTLINE_WIDTH_MIN} and "
            f"{RENDER_OUTLINE_WIDTH_MAX}"
        )
    return result


def canonical_render_preferred_size(
    value: Any,
    *,
    field_name: str = "preferred_size",
) -> float:
    """Return one exact renderer-supported preferred em target in pixels."""

    return _require_number(
        value,
        field_name,
        minimum=RENDER_PREFERRED_SIZE_MIN,
        maximum=RENDER_PREFERRED_SIZE_MAX,
    )


def canonical_render_shadow_blur(
    value: Any,
    *,
    field_name: str = "shadow_blur",
) -> float:
    """Return one exact renderer-supported shadow blur radius in pixels."""

    return _require_number(
        value,
        field_name,
        minimum=RENDER_SHADOW_BLUR_MIN,
        maximum=RENDER_SHADOW_BLUR_MAX,
    )


def canonical_render_shadow_offset(
    value: Any,
    *,
    field_name: str = "shadow_offset",
) -> tuple[float, float]:
    """Return one exact renderer-supported shadow offset in X/Y pixels."""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        raise ValueError(f"{field_name} must contain exactly x and y")
    return (
        _require_number(
            value[0],
            f"{field_name}.x",
            minimum=RENDER_SHADOW_OFFSET_MIN,
            maximum=RENDER_SHADOW_OFFSET_MAX,
        ),
        _require_number(
            value[1],
            f"{field_name}.y",
            minimum=RENDER_SHADOW_OFFSET_MIN,
            maximum=RENDER_SHADOW_OFFSET_MAX,
        ),
    )


def canonical_render_box(
    value: Any,
    *,
    field_name: str = "render_box",
) -> tuple[int, int, int, int]:
    """Return one exact integer X/Y/width/height target box."""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 4
    ):
        raise ValueError(
            f"{field_name} must contain exactly x, y, width, and height"
        )
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{field_name}[{index}] must be an exact integer")
        result.append(item)
    if result[2] <= 0 or result[3] <= 0:
        raise ValueError(f"{field_name} width and height must be positive")
    return result[0], result[1], result[2], result[3]


def canonical_render_font_role(
    value: Any,
    *,
    field_name: str = "font_role",
) -> str:
    """Return one exact renderer-supported registered CJK font role."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an exact registered role ID")
    if value != value.strip() or not value:
        raise ValueError(f"{field_name} must be an exact registered role ID")
    if value not in REGISTERED_RENDER_FONT_ROLES:
        raise ValueError(f"{field_name} is not a registered renderer font role")
    return value


def canonical_render_font_weight_tier(
    value: Any,
    *,
    field_name: str = "font_weight_tier",
) -> str:
    """Return one exact renderer-supported registered weight tier."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an exact font-weight tier")
    if value != value.strip() or not value:
        raise ValueError(f"{field_name} must be an exact font-weight tier")
    if value not in RENDER_FONT_WEIGHT_TIERS:
        raise ValueError(
            f"{field_name} must be slender, base, emphasis, or heavy"
        )
    return value


def _validate_color(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a hex color string")
    if len(value) not in {7, 9} or not value.startswith("#"):
        raise ValueError(f"{field_name} must use #RRGGBB or #RRGGBBAA")
    if any(character not in "0123456789abcdefABCDEF" for character in value[1:]):
        raise ValueError(f"{field_name} must use hexadecimal digits")


def _validate_style_field(field: str, value: Any) -> None:
    if field == "font_role":
        canonical_render_font_role(value, field_name="font_role")
    elif field == "font_weight_tier":
        canonical_render_font_weight_tier(
            value,
            field_name="font_weight_tier",
        )
    elif field in {"font_family", "font_face"}:
        _require_non_empty(value, f"payload.fields.{field}")
    elif field == "font_weight":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("font_weight must be an integer")
        if value < 1 or value > 1000:
            raise ValueError("font_weight must be between 1 and 1000")
    elif field == "preferred_size":
        canonical_render_preferred_size(value)
    elif field in {"minimum_size", "maximum_size"}:
        _require_number(value, field, minimum=0.1, maximum=2048.0)
    elif field in {"fill_color", "outline_color"}:
        _validate_color(value, field)
    elif field == "shadow_color":
        canonical_render_shadow_color(value)
    elif field == "outline_width":
        canonical_render_outline_width(value)
    elif field in {"shadow_blur", "shadow_spread"}:
        _require_number(value, field, minimum=0.0, maximum=128.0)
    elif field == "shadow_enabled":
        if not isinstance(value, bool):
            raise TypeError("shadow_enabled must be a boolean")
    elif field == "shadow_offset":
        canonical_render_shadow_offset(value)
    elif field == "shadow_opacity":
        _require_number(value, field, minimum=0.0, maximum=1.0)


def _validate_layout_field(field: str, value: Any) -> None:
    if field in {"x", "y"}:
        _require_number(value, field, minimum=-100000, maximum=100000)
    elif field in {"width", "height"}:
        _require_number(value, field, minimum=0.1, maximum=100000)
    elif field == "rotation":
        canonical_render_rotation(value, field_name=field)
    elif field == "writing_mode":
        if value not in CANONICAL_WRITING_MODES:
            raise ValueError("writing_mode is unsupported")
    elif field == "alignment":
        if value not in {"start", "center", "end", "left", "right", "justify"}:
            raise ValueError("alignment is unsupported")
    elif field == "line_height":
        canonical_render_line_height(value, field_name=field)
    elif field in {"letter_spacing", "column_spacing", "run_spacing"}:
        _require_number(value, field, minimum=-256, maximum=1024)
    elif field == "render_box":
        canonical_render_box(value)
    elif field == "break_hints":
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
        ):
            raise TypeError("break_hints must be a list of character offsets")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
            raise ValueError("break_hints must contain non-negative integers")


def validate_edit_payload(
    domain: EditDomain,
    operation: str,
    payload: Mapping[str, Any],
    *,
    allow_unsupported_fill_color: bool = False,
) -> None:
    operation = _require_non_empty(operation, "operation")
    if domain is EditDomain.STRUCTURAL:
        if operation in {"exclude", "restore"}:
            _require_exact_keys(payload, required=frozenset())
        elif operation == "add_user_parent":
            _require_exact_keys(
                payload,
                required=frozenset(
                    {
                        "identity_namespace",
                        "root_id",
                        "root_identity_namespace",
                        "role",
                        "workflow_area_bbox",
                        "canvas_size",
                        "order_policy",
                    }
                ),
            )
            if payload.get("identity_namespace") != USER_PARENT_IDENTITY_NAMESPACE:
                raise ValueError(
                    "add_user_parent identity_namespace must be user_parent_v1"
                )
            if payload.get("root_identity_namespace") != USER_ROOT_IDENTITY_NAMESPACE:
                raise ValueError(
                    "add_user_parent root_identity_namespace must be user_root_v1"
                )
            user_root_identity_suffix(str(payload.get("root_id") or ""))
            if payload.get("role") not in {"speech", "caption"}:
                raise ValueError("add_user_parent role must be speech or caption")
            if payload.get("order_policy") != "append":
                raise ValueError("add_user_parent order_policy must be append")
            bbox = payload.get("workflow_area_bbox")
            if (
                not isinstance(bbox, Sequence)
                or isinstance(bbox, (str, bytes, bytearray))
                or len(bbox) != 4
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in bbox
                )
            ):
                raise ValueError(
                    "workflow_area_bbox must contain exact integer x, y, width, height"
                )
            if int(bbox[0]) < 0 or int(bbox[1]) < 0:
                raise ValueError("workflow_area_bbox origin must not be negative")
            if int(bbox[2]) <= 0 or int(bbox[3]) <= 0:
                raise ValueError(
                    "workflow_area_bbox width and height must be positive"
                )
            canvas_size = payload.get("canvas_size")
            if (
                not isinstance(canvas_size, Sequence)
                or isinstance(canvas_size, (str, bytes, bytearray))
                or len(canvas_size) != 2
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item <= 0
                    for item in canvas_size
                )
            ):
                raise ValueError(
                    "canvas_size must contain positive exact integer width and height"
                )
            page_width, page_height = int(canvas_size[0]), int(canvas_size[1])
            if page_width * page_height > 50_000_000:
                raise ValueError("canvas_size exceeds the safety limit")
            if (
                int(bbox[0]) + int(bbox[2]) > page_width
                or int(bbox[1]) + int(bbox[3]) > page_height
            ):
                raise ValueError("workflow_area_bbox must remain inside canvas_size")
        elif operation == "split_user_parent":
            _require_exact_keys(
                payload,
                required=frozenset(
                    {
                        "identity_namespace",
                        "root_identity_namespace",
                        "source_root_id",
                        "source_authored_edit_id",
                        "source_role",
                        "source_workflow_area_bbox",
                        "canvas_size",
                        "orientation",
                        "split_offset",
                        "child_parent_ids",
                        "child_root_ids",
                        "child_workflow_area_bboxes",
                        "order_policy",
                    }
                ),
                optional=frozenset({"child_source_evidence_mappings"}),
            )
            if payload.get("identity_namespace") != USER_PARENT_IDENTITY_NAMESPACE:
                raise ValueError(
                    "split_user_parent identity_namespace must be user_parent_v1"
                )
            if payload.get("root_identity_namespace") != USER_ROOT_IDENTITY_NAMESPACE:
                raise ValueError(
                    "split_user_parent root_identity_namespace must be user_root_v1"
                )
            user_root_identity_suffix(str(payload.get("source_root_id") or ""))
            _require_non_empty(
                payload.get("source_authored_edit_id"),
                "payload.source_authored_edit_id",
            )
            if payload.get("source_role") not in {"speech", "caption"}:
                raise ValueError(
                    "split_user_parent source_role must be speech or caption"
                )
            if payload.get("orientation") not in {"vertical", "horizontal"}:
                raise ValueError(
                    "split_user_parent orientation must be vertical or horizontal"
                )
            split_offset = payload.get("split_offset")
            if isinstance(split_offset, bool) or not isinstance(split_offset, int):
                raise ValueError("split_user_parent split_offset must be an integer")
            if payload.get("order_policy") != "replace_source":
                raise ValueError(
                    "split_user_parent order_policy must be replace_source"
                )
            source_bbox = payload.get("source_workflow_area_bbox")
            if (
                not isinstance(source_bbox, Sequence)
                or isinstance(source_bbox, (str, bytes, bytearray))
                or len(source_bbox) != 4
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in source_bbox
                )
            ):
                raise ValueError(
                    "source_workflow_area_bbox must contain exact integer x, y, width, height"
                )
            source_x, source_y, source_width, source_height = (
                int(item) for item in source_bbox
            )
            if source_x < 0 or source_y < 0 or source_width <= 0 or source_height <= 0:
                raise ValueError("source_workflow_area_bbox is invalid")
            canvas_size = payload.get("canvas_size")
            if (
                not isinstance(canvas_size, Sequence)
                or isinstance(canvas_size, (str, bytes, bytearray))
                or len(canvas_size) != 2
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item <= 0
                    for item in canvas_size
                )
            ):
                raise ValueError(
                    "split_user_parent canvas_size must contain positive exact integers"
                )
            page_width, page_height = (int(item) for item in canvas_size)
            if page_width * page_height > 50_000_000:
                raise ValueError("split_user_parent canvas_size exceeds the safety limit")
            if (
                source_x + source_width > page_width
                or source_y + source_height > page_height
            ):
                raise ValueError(
                    "source_workflow_area_bbox must remain inside canvas_size"
                )
            child_parent_ids = payload.get("child_parent_ids")
            child_root_ids = payload.get("child_root_ids")
            for field_name, values in (
                ("child_parent_ids", child_parent_ids),
                ("child_root_ids", child_root_ids),
            ):
                if (
                    not isinstance(values, Sequence)
                    or isinstance(values, (str, bytes, bytearray))
                    or len(values) != 2
                    or any(not isinstance(value, str) or not value for value in values)
                    or len(set(values)) != 2
                ):
                    raise ValueError(f"{field_name} must contain two unique identities")
            assert isinstance(child_parent_ids, Sequence)
            assert isinstance(child_root_ids, Sequence)
            for child_parent_id, child_root_id in zip(
                child_parent_ids,
                child_root_ids,
            ):
                validate_user_parent_identity_pair(
                    str(child_parent_id),
                    str(child_root_id),
                )
            child_bboxes = payload.get("child_workflow_area_bboxes")
            if (
                not isinstance(child_bboxes, Sequence)
                or isinstance(child_bboxes, (str, bytes, bytearray))
                or len(child_bboxes) != 2
            ):
                raise ValueError(
                    "child_workflow_area_bboxes must contain two exact bboxes"
                )
            normalized_child_bboxes: list[tuple[int, int, int, int]] = []
            for index, child_bbox in enumerate(child_bboxes):
                if (
                    not isinstance(child_bbox, Sequence)
                    or isinstance(child_bbox, (str, bytes, bytearray))
                    or len(child_bbox) != 4
                    or any(
                        isinstance(item, bool) or not isinstance(item, int)
                        for item in child_bbox
                    )
                ):
                    raise ValueError(
                        f"child_workflow_area_bboxes[{index}] must be an exact integer bbox"
                    )
                normalized_child_bboxes.append(
                    tuple(int(item) for item in child_bbox)
                )
            if payload.get("orientation") == "vertical":
                if split_offset <= 0 or split_offset >= source_width:
                    raise ValueError(
                        "vertical split_offset must lie strictly inside source width"
                    )
                expected_child_bboxes = (
                    (source_x, source_y, split_offset, source_height),
                    (
                        source_x + split_offset,
                        source_y,
                        source_width - split_offset,
                        source_height,
                    ),
                )
            else:
                if split_offset <= 0 or split_offset >= source_height:
                    raise ValueError(
                        "horizontal split_offset must lie strictly inside source height"
                    )
                expected_child_bboxes = (
                    (source_x, source_y, source_width, split_offset),
                    (
                        source_x,
                        source_y + split_offset,
                        source_width,
                        source_height - split_offset,
                    ),
                )
            if tuple(normalized_child_bboxes) != expected_child_bboxes:
                raise ValueError(
                    "split child bboxes must exactly partition the source bbox"
                )
            child_mapping_values = payload.get("child_source_evidence_mappings")
            if child_mapping_values is not None:
                if (
                    not isinstance(child_mapping_values, Sequence)
                    or isinstance(child_mapping_values, (str, bytes, bytearray))
                    or len(child_mapping_values) != 2
                    or any(
                        not isinstance(value, Mapping)
                        for value in child_mapping_values
                    )
                ):
                    raise ValueError(
                        "child_source_evidence_mappings must contain two mappings"
                    )
                child_mappings = tuple(
                    ParentSourceEvidenceMappingV1.from_dict(value)
                    for value in child_mapping_values
                )
                if any(
                    mapping.workflow_bbox[0] < expected_child_bboxes[index][0]
                    or mapping.workflow_bbox[1] < expected_child_bboxes[index][1]
                    or mapping.workflow_bbox[0] + mapping.workflow_bbox[2]
                    > expected_child_bboxes[index][0]
                    + expected_child_bboxes[index][2]
                    or mapping.workflow_bbox[1] + mapping.workflow_bbox[3]
                    > expected_child_bboxes[index][1]
                    + expected_child_bboxes[index][3]
                    or any(role != payload.get("source_role") for role in mapping.source_roles)
                    for index, mapping in enumerate(child_mappings)
                ):
                    raise ValueError(
                        "split child source evidence must remain inside its child scope"
                    )
                combined_parent_ids = tuple(
                    parent_id
                    for mapping in child_mappings
                    for parent_id in mapping.source_parent_ids
                )
                combined_bundle_ids = tuple(
                    bundle_id
                    for mapping in child_mappings
                    for bundle_id in mapping.source_bundle_ids
                )
                if (
                    len(set(combined_parent_ids)) != len(combined_parent_ids)
                    or len(set(combined_bundle_ids)) != len(combined_bundle_ids)
                ):
                    raise ValueError(
                        "split child source-evidence mappings must not overlap"
                    )
        elif operation == "merge_pipeline_parents":
            _require_exact_keys(
                payload,
                required=frozenset(
                    {
                        "identity_namespace",
                        "root_identity_namespace",
                        "merged_root_id",
                        "source_parent_ids",
                        "source_root_ids",
                        "source_automatic_fingerprints",
                        "source_bboxes",
                        "source_texts",
                        "source_text_fingerprints",
                        "source_role",
                        "merged_workflow_area_bbox",
                        "merged_source_text",
                        "canvas_size",
                        "predecessor_ordered_parent_ids",
                        "order_policy",
                    }
                ),
            )
            if payload.get("identity_namespace") != USER_PARENT_IDENTITY_NAMESPACE:
                raise ValueError(
                    "merge_pipeline_parents identity_namespace must be user_parent_v1"
                )
            if payload.get("root_identity_namespace") != USER_ROOT_IDENTITY_NAMESPACE:
                raise ValueError(
                    "merge_pipeline_parents root_identity_namespace must be user_root_v1"
                )
            user_root_identity_suffix(str(payload.get("merged_root_id") or ""))
            if payload.get("source_role") not in {"speech", "caption"}:
                raise ValueError(
                    "merge_pipeline_parents source_role must be speech or caption"
                )
            if payload.get("order_policy") != "replace_sources":
                raise ValueError(
                    "merge_pipeline_parents order_policy must be replace_sources"
                )

            source_parent_ids = payload.get("source_parent_ids")
            source_root_ids = payload.get("source_root_ids")
            source_automatic_fingerprints = payload.get(
                "source_automatic_fingerprints"
            )
            source_texts = payload.get("source_texts")
            source_text_fingerprints = payload.get("source_text_fingerprints")
            for field_name, values in (
                ("source_parent_ids", source_parent_ids),
                ("source_root_ids", source_root_ids),
                ("source_automatic_fingerprints", source_automatic_fingerprints),
                ("source_texts", source_texts),
                ("source_text_fingerprints", source_text_fingerprints),
            ):
                if (
                    not isinstance(values, Sequence)
                    or isinstance(values, (str, bytes, bytearray))
                    or len(values) != 2
                ):
                    raise ValueError(f"{field_name} must contain exactly two values")
            assert isinstance(source_parent_ids, Sequence)
            assert isinstance(source_root_ids, Sequence)
            assert isinstance(source_automatic_fingerprints, Sequence)
            assert isinstance(source_texts, Sequence)
            assert isinstance(source_text_fingerprints, Sequence)
            if (
                any(not isinstance(value, str) or not value for value in source_parent_ids)
                or len(set(source_parent_ids)) != 2
            ):
                raise ValueError("source_parent_ids must contain two unique identities")
            if any(
                not isinstance(value, str) or not value for value in source_root_ids
            ):
                raise ValueError("source_root_ids must contain two non-empty identities")
            for index, fingerprint in enumerate(source_automatic_fingerprints):
                _require_sha256(
                    fingerprint,
                    f"source_automatic_fingerprints[{index}]",
                )
            for index, text in enumerate(source_texts):
                if not isinstance(text, str) or not text or not text.strip():
                    raise ValueError(
                        f"source_texts[{index}] must contain exact non-empty OCR text"
                    )
            for index, fingerprint in enumerate(source_text_fingerprints):
                _require_sha256(
                    fingerprint,
                    f"source_text_fingerprints[{index}]",
                )
            if payload.get("merged_source_text") != "".join(source_texts):
                raise ValueError(
                    "merged_source_text must exactly concatenate source_texts"
                )

            canvas_size = payload.get("canvas_size")
            if (
                not isinstance(canvas_size, Sequence)
                or isinstance(canvas_size, (str, bytes, bytearray))
                or len(canvas_size) != 2
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item <= 0
                    for item in canvas_size
                )
            ):
                raise ValueError(
                    "merge_pipeline_parents canvas_size must contain positive exact integers"
                )
            page_width, page_height = (int(item) for item in canvas_size)
            if page_width * page_height > 50_000_000:
                raise ValueError(
                    "merge_pipeline_parents canvas_size exceeds the safety limit"
                )
            source_bboxes = payload.get("source_bboxes")
            if (
                not isinstance(source_bboxes, Sequence)
                or isinstance(source_bboxes, (str, bytes, bytearray))
                or len(source_bboxes) != 2
            ):
                raise ValueError("source_bboxes must contain two exact bboxes")
            normalized_source_bboxes: list[tuple[int, int, int, int]] = []
            for index, source_bbox in enumerate(source_bboxes):
                if (
                    not isinstance(source_bbox, Sequence)
                    or isinstance(source_bbox, (str, bytes, bytearray))
                    or len(source_bbox) != 4
                    or any(
                        isinstance(item, bool) or not isinstance(item, int)
                        for item in source_bbox
                    )
                ):
                    raise ValueError(
                        f"source_bboxes[{index}] must be an exact integer bbox"
                    )
                bbox = tuple(int(item) for item in source_bbox)
                if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValueError(f"source_bboxes[{index}] is invalid")
                if bbox[0] + bbox[2] > page_width or bbox[1] + bbox[3] > page_height:
                    raise ValueError(
                        f"source_bboxes[{index}] must remain inside canvas_size"
                    )
                normalized_source_bboxes.append(bbox)
            left = min(bbox[0] for bbox in normalized_source_bboxes)
            top = min(bbox[1] for bbox in normalized_source_bboxes)
            right = max(bbox[0] + bbox[2] for bbox in normalized_source_bboxes)
            bottom = max(bbox[1] + bbox[3] for bbox in normalized_source_bboxes)
            expected_merged_bbox = (left, top, right - left, bottom - top)
            merged_bbox = payload.get("merged_workflow_area_bbox")
            if (
                not isinstance(merged_bbox, Sequence)
                or isinstance(merged_bbox, (str, bytes, bytearray))
                or len(merged_bbox) != 4
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in merged_bbox
                )
                or tuple(int(item) for item in merged_bbox) != expected_merged_bbox
            ):
                raise ValueError(
                    "merged_workflow_area_bbox must be the exact source bbox union"
                )

            predecessor_order = payload.get("predecessor_ordered_parent_ids")
            if (
                not isinstance(predecessor_order, Sequence)
                or isinstance(predecessor_order, (str, bytes, bytearray))
                or len(predecessor_order) < 2
                or any(
                    not isinstance(parent_id, str) or not parent_id
                    for parent_id in predecessor_order
                )
                or len(set(predecessor_order)) != len(predecessor_order)
            ):
                raise ValueError(
                    "predecessor_ordered_parent_ids must be a unique identity sequence"
                )
            first_index = tuple(predecessor_order).index(source_parent_ids[0])
            if (
                first_index + 1 >= len(predecessor_order)
                or predecessor_order[first_index + 1] != source_parent_ids[1]
            ):
                raise ValueError(
                    "source_parent_ids must be consecutive in predecessor order"
                )
        elif operation == "set_geometry":
            _require_exact_keys(
                payload,
                required=frozenset({"bbox", "canvas_size"}),
            )
            bbox = payload.get("bbox")
            if (
                not isinstance(bbox, Sequence)
                or isinstance(bbox, (str, bytes, bytearray))
                or len(bbox) != 4
            ):
                raise ValueError("structural bbox must contain x, y, width, height")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in bbox):
                raise ValueError("structural bbox values must be exact integers")
            if int(bbox[0]) < 0 or int(bbox[1]) < 0:
                raise ValueError("structural bbox origin must not be negative")
            if int(bbox[2]) <= 0 or int(bbox[3]) <= 0:
                raise ValueError("structural bbox width and height must be positive")
            canvas_size = payload.get("canvas_size")
            if (
                not isinstance(canvas_size, Sequence)
                or isinstance(canvas_size, (str, bytes, bytearray))
                or len(canvas_size) != 2
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item <= 0
                    for item in canvas_size
                )
            ):
                raise ValueError(
                    "structural canvas_size must contain positive integer width and height"
                )
            page_width, page_height = (int(canvas_size[0]), int(canvas_size[1]))
            if page_width * page_height > 50_000_000:
                raise ValueError("structural canvas_size exceeds the safety limit")
            if (
                int(bbox[0]) + int(bbox[2]) > page_width
                or int(bbox[1]) + int(bbox[3]) > page_height
            ):
                raise ValueError("structural bbox must remain inside canvas_size")
        elif operation == "set_reading_order":
            _require_exact_keys(
                payload,
                required=frozenset(
                    {"selected_parent_id", "ordered_parent_ids"}
                ),
            )
            _require_non_empty(
                payload.get("selected_parent_id"),
                "payload.selected_parent_id",
            )
            ordered_parent_ids = payload.get("ordered_parent_ids")
            if (
                not isinstance(ordered_parent_ids, Sequence)
                or isinstance(ordered_parent_ids, (str, bytes, bytearray))
                or not ordered_parent_ids
            ):
                raise TypeError(
                    "ordered_parent_ids must be a non-empty parent-ID sequence"
                )
            normalized_parent_ids: list[str] = []
            for index, parent_id in enumerate(ordered_parent_ids):
                if not isinstance(parent_id, str):
                    raise TypeError(
                        f"ordered_parent_ids[{index}] must be a string"
                    )
                normalized_parent_ids.append(
                    _require_non_empty(
                        parent_id,
                        f"payload.ordered_parent_ids[{index}]",
                    )
                )
            if len(set(normalized_parent_ids)) != len(normalized_parent_ids):
                raise ValueError("ordered_parent_ids must not contain duplicates")
            if payload.get("selected_parent_id") not in normalized_parent_ids:
                raise ValueError(
                    "selected_parent_id must occur in ordered_parent_ids"
                )
        elif operation == "set_role":
            _require_exact_keys(payload, required=frozenset({"role"}))
            _require_non_empty(payload.get("role"), "payload.role")
        else:
            raise ValueError(
                f"unsupported GUI-1 structural operation: {operation}"
            )
        return

    if domain is EditDomain.SOURCE_TEXT:
        if operation == "replace":
            _require_exact_keys(
                payload,
                required=frozenset({"text"}),
                optional=frozenset({"revision_base"}),
            )
            if not isinstance(payload.get("text"), str):
                raise ValueError("text replacement must be a string")
            if "revision_base" in payload:
                SourceTextRevisionBaseV1.from_dict(payload.get("revision_base"))
        elif operation == "select_revision":
            _require_exact_keys(payload, required=frozenset({"revision_id"}))
            revision_id = _require_non_empty(
                payload.get("revision_id"),
                "payload.revision_id",
            )
            if not revision_id.startswith("ocr-source-revision-v1-"):
                raise ValueError("source revision identity is invalid")
        elif operation == "restore_automatic":
            _require_exact_keys(payload, required=frozenset())
        elif operation == "restore_selected_revision":
            _require_exact_keys(
                payload,
                required=frozenset({"revision_base"}),
            )
            SourceTextRevisionBaseV1.from_dict(payload.get("revision_base"))
        else:
            raise ValueError(f"unsupported source_text operation: {operation}")
        return

    if domain is EditDomain.TARGET_TEXT:
        if operation == "replace":
            _require_exact_keys(
                payload,
                required=frozenset({"text", "source_fingerprint"}),
                optional=frozenset({"revision_base", "source_evidence_base"}),
            )
            if not isinstance(payload.get("text"), str):
                raise ValueError("text replacement must be a string")
            _require_sha256(
                payload.get("source_fingerprint"),
                "payload.source_fingerprint",
            )
            if "revision_base" in payload:
                base = TargetTextRevisionBaseV1.from_dict(
                    payload.get("revision_base")
                )
                if base.source_fingerprint != payload.get("source_fingerprint"):
                    raise ValueError(
                        "target replacement source and revision base differ"
                    )
            if "source_evidence_base" in payload:
                if "revision_base" in payload:
                    raise ValueError(
                        "target replacement may carry only one immutable base"
                    )
                base = ParentSourceEvidenceMappingV1.from_dict(
                    payload.get("source_evidence_base")
                )
                if not base.source_text:
                    raise ValueError(
                        "target replacement source-evidence base has no OCR text"
                    )
        elif operation == "select_revision":
            _require_exact_keys(
                payload,
                required=frozenset({"revision_id", "source_fingerprint"}),
            )
            revision_id = _require_non_empty(
                payload.get("revision_id"),
                "payload.revision_id",
            )
            if not revision_id.startswith("translation-revision-v1-"):
                raise ValueError("translation revision identity is invalid")
            _require_sha256(
                payload.get("source_fingerprint"),
                "payload.source_fingerprint",
            )
        elif operation == "restore_automatic":
            _require_exact_keys(payload, required=frozenset())
        elif operation == "restore_selected_revision":
            _require_exact_keys(
                payload,
                required=frozenset({"revision_base"}),
            )
            TargetTextRevisionBaseV1.from_dict(payload.get("revision_base"))
        elif operation == "restore_mapped_pipeline":
            _require_exact_keys(
                payload,
                required=frozenset({"source_evidence_base"}),
            )
            base = ParentSourceEvidenceMappingV1.from_dict(
                payload.get("source_evidence_base")
            )
            if base.target_text is None:
                raise ValueError(
                    "mapped pipeline target evidence is unavailable"
                )
        else:
            raise ValueError(f"unsupported target_text operation: {operation}")
        return

    if domain is EditDomain.CLEANUP:
        if operation != "select_revision":
            raise ValueError(f"unsupported cleanup operation: {operation}")
        _require_exact_keys(payload, required=frozenset({"revision_id"}))
        _require_non_empty(payload.get("revision_id"), "payload.revision_id")
        return

    if domain is EditDomain.RENDER_STYLE:
        if operation == "set_fields":
            _require_fields_mapping(payload, allowed=_STYLE_FIELDS)
            field, value = next(iter(payload["fields"].items()))
            if (
                str(field) in {"fill_color", "outline_color"}
                and not allow_unsupported_fill_color
            ):
                canonicalizer = (
                    canonical_render_fill_color
                    if str(field) == "fill_color"
                    else canonical_render_outline_color
                )
                canonicalizer(value, field_name=f"payload.fields.{field}")
            else:
                _validate_style_field(str(field), value)
        elif operation == "restore_automatic":
            _require_exact_keys(
                payload,
                required=frozenset({"fields"}),
            )
            fields = payload.get("fields")
            if (
                not isinstance(fields, Sequence)
                or isinstance(fields, (str, bytes, bytearray))
                or len(fields) != 1
            ):
                raise ValueError("restore edits must name exactly one field")
            unknown = frozenset(str(field) for field in fields) - _STYLE_FIELDS
            if unknown:
                raise ValueError(
                    f"unsupported restore fields: {sorted(unknown)}"
                )
        else:
            raise ValueError(f"unsupported render_style operation: {operation}")
        return

    if domain is EditDomain.RENDER_LAYOUT:
        if operation == "set_fields":
            _require_fields_mapping(payload, allowed=_LAYOUT_FIELDS)
            field, value = next(iter(payload["fields"].items()))
            _validate_layout_field(str(field), value)
        elif operation == "restore_automatic":
            _require_exact_keys(
                payload,
                required=frozenset({"fields"}),
            )
            fields = payload.get("fields")
            if (
                not isinstance(fields, Sequence)
                or isinstance(fields, (str, bytes, bytearray))
                or len(fields) != 1
            ):
                raise ValueError("restore edits must name exactly one field")
            unknown = frozenset(str(field) for field in fields) - _LAYOUT_FIELDS
            if unknown:
                raise ValueError(
                    f"unsupported restore fields: {sorted(unknown)}"
                )
        else:
            raise ValueError(f"unsupported render_layout operation: {operation}")
        return

    if domain is EditDomain.REVIEW_METADATA:
        if operation != "set_fields":
            raise ValueError(f"unsupported review_metadata operation: {operation}")
        _require_exact_keys(payload, required=frozenset({"fields"}))
        fields = payload.get("fields")
        if not isinstance(fields, Mapping) or not fields:
            raise ValueError("review metadata fields must be a non-empty mapping")
        if len(fields) != 1:
            raise ValueError(
                "review metadata edits must contain exactly one field"
            )
        unknown = frozenset(str(key) for key in fields) - {"needs_review", "note"}
        if unknown:
            raise ValueError(f"unsupported review metadata fields: {sorted(unknown)}")
        if "needs_review" in fields and not isinstance(fields["needs_review"], bool):
            raise TypeError("review metadata needs_review must be a boolean")
        if "note" in fields and not isinstance(fields["note"], str):
            raise TypeError("review metadata note must be a string")
        return

    if domain is EditDomain.GLOSSARY:
        if operation == "set_entry":
            _require_exact_keys(payload, required=frozenset({"entry"}))
            entry = payload.get("entry")
            if not isinstance(entry, Mapping):
                raise ValueError("glossary entry must be a mapping")
            required = {"entry_id", "source", "target"}
            optional = {"notes", "aliases", "priority"}
            missing = required - set(entry)
            unknown = set(entry) - required - optional
            if missing:
                raise ValueError(
                    f"glossary entry is missing fields: {sorted(missing)}"
                )
            if unknown:
                raise ValueError(
                    "glossary entry has unsupported fields: "
                    f"{sorted(unknown)}"
                )
            _require_non_empty(entry.get("entry_id"), "payload.entry.entry_id")
            _require_non_empty(entry.get("source"), "payload.entry.source")
            if not isinstance(entry.get("target"), str):
                raise ValueError("payload.entry.target must be a string")
            if "notes" in entry and not isinstance(entry.get("notes"), str):
                raise TypeError("payload.entry.notes must be a string")
            aliases = entry.get("aliases", ())
            if (
                not isinstance(aliases, Sequence)
                or isinstance(aliases, (str, bytes, bytearray))
                or any(not isinstance(alias, str) for alias in aliases)
            ):
                raise TypeError("payload.entry.aliases must be a list of strings")
            priority = entry.get("priority", "soft")
            if isinstance(priority, str):
                if priority not in {"soft", "hard"}:
                    raise ValueError(
                        "payload.entry.priority must be 'soft' or 'hard'"
                    )
            elif isinstance(priority, bool) or not isinstance(priority, int):
                raise TypeError(
                    "payload.entry.priority must be an integer or "
                    "'soft'/'hard'"
                )
        elif operation == "remove_entry":
            _require_exact_keys(payload, required=frozenset({"entry_id"}))
            _require_non_empty(payload.get("entry_id"), "payload.entry_id")
        else:
            raise ValueError(f"unsupported glossary operation: {operation}")
        return

    if domain is EditDomain.LEDGER_CONTROL:
        if operation not in {"revoke", "reapply"}:
            raise ValueError(f"unsupported ledger control operation: {operation}")
        _require_exact_keys(payload, required=frozenset({"edit_id"}))
        _require_non_empty(payload.get("edit_id"), "payload.edit_id")
        return

    raise ValueError(f"unsupported edit domain: {domain.value}")


@dataclass(frozen=True)
class EditTarget:
    kind: EditTargetKind
    parent_id: str = ""
    artifact_id: str = ""
    edit_id: str = ""

    def __post_init__(self) -> None:
        kind = EditTargetKind(self.kind)
        object.__setattr__(self, "kind", kind)
        identifiers = {
            EditTargetKind.PARENT: self.parent_id,
            EditTargetKind.ARTIFACT: self.artifact_id,
            EditTargetKind.EDIT: self.edit_id,
        }
        required = identifiers.get(kind)
        if required is not None:
            _require_non_empty(required, f"target.{kind.value}_id")
        populated = {
            "parent_id": bool(self.parent_id),
            "artifact_id": bool(self.artifact_id),
            "edit_id": bool(self.edit_id),
        }
        allowed = {
            EditTargetKind.PROJECT: set(),
            EditTargetKind.PAGE: set(),
            EditTargetKind.PARENT: {"parent_id"},
            EditTargetKind.ARTIFACT: {"artifact_id"},
            EditTargetKind.EDIT: {"edit_id"},
        }[kind]
        unexpected = {name for name, present in populated.items() if present and name not in allowed}
        if unexpected:
            raise ValueError(f"target has unexpected identifiers: {sorted(unexpected)}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind.value}
        if self.parent_id:
            result["parent_id"] = self.parent_id
        if self.artifact_id:
            result["artifact_id"] = self.artifact_id
        if self.edit_id:
            result["edit_id"] = self.edit_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EditTarget":
        if not isinstance(value, Mapping):
            raise TypeError("edit target must be a mapping")
        unknown = frozenset(value) - {"kind", "parent_id", "artifact_id", "edit_id"}
        if unknown:
            raise ValueError(f"edit target has unsupported fields: {sorted(unknown)}")
        return cls(
            kind=EditTargetKind(str(value.get("kind") or "")),
            parent_id=str(value.get("parent_id") or ""),
            artifact_id=str(value.get("artifact_id") or ""),
            edit_id=str(value.get("edit_id") or ""),
        )


@dataclass(frozen=True)
class ProjectEdit:
    edit_id: str
    project_id: str
    page_id: str
    target: EditTarget
    domain: EditDomain
    operation: str
    payload: Mapping[str, Any]
    base_revision_id: str
    base_fingerprint: str
    supersedes_edit_id: str | None
    provenance: str
    created_at: str
    active: bool = True
    edit_schema_version: str = EDIT_SCHEMA_VERSION
    _allow_unsupported_fill_color: InitVar[bool] = False

    def __post_init__(self, _allow_unsupported_fill_color: bool) -> None:
        if self.edit_schema_version != EDIT_SCHEMA_VERSION:
            raise ValueError(f"unsupported edit schema: {self.edit_schema_version}")
        object.__setattr__(self, "edit_id", _require_non_empty(self.edit_id, "edit_id"))
        object.__setattr__(self, "project_id", _require_non_empty(self.project_id, "project_id"))
        object.__setattr__(self, "page_id", _require_non_empty(self.page_id, "page_id"))
        target = self.target if isinstance(self.target, EditTarget) else EditTarget.from_dict(self.target)
        object.__setattr__(self, "target", target)
        domain = EditDomain(self.domain)
        object.__setattr__(self, "domain", domain)
        operation = _require_non_empty(self.operation, "operation")
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.payload, Mapping):
            raise TypeError("edit payload must be a mapping")
        frozen_payload = _freeze_json(self.payload)
        object.__setattr__(self, "payload", frozen_payload)
        validate_edit_payload(
            domain,
            operation,
            frozen_payload,
            allow_unsupported_fill_color=_allow_unsupported_fill_color,
        )
        object.__setattr__(self, "base_revision_id", _require_non_empty(self.base_revision_id, "base_revision_id"))
        object.__setattr__(self, "base_fingerprint", _require_sha256(self.base_fingerprint, "base_fingerprint"))
        if self.supersedes_edit_id is not None:
            object.__setattr__(
                self,
                "supersedes_edit_id",
                _require_non_empty(self.supersedes_edit_id, "supersedes_edit_id"),
            )
            if self.supersedes_edit_id == self.edit_id:
                raise ValueError("an edit cannot supersede itself")
        provenance = _require_non_empty(self.provenance, "provenance")
        if provenance != "user":
            raise ValueError(f"unsupported edit provenance: {provenance}")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "created_at", _validate_created_at(self.created_at))
        if not isinstance(self.active, bool):
            raise TypeError("active must be a boolean")
        if self.active is not True:
            raise ValueError(
                "stored edits are append-only events; revoke through a control record"
            )
        if domain is EditDomain.LEDGER_CONTROL:
            if target.kind is not EditTargetKind.EDIT:
                raise ValueError("ledger controls must target an edit")
            if str(frozen_payload.get("edit_id") or "") != target.edit_id:
                raise ValueError("ledger control target and payload edit IDs differ")
        elif target.kind is EditTargetKind.EDIT:
            raise ValueError("only ledger controls may target an edit")
        allowed_targets = {
            EditDomain.STRUCTURAL: {
                EditTargetKind.PAGE,
                EditTargetKind.PARENT,
            },
            EditDomain.SOURCE_TEXT: {EditTargetKind.PARENT},
            EditDomain.TARGET_TEXT: {EditTargetKind.PARENT},
            EditDomain.CLEANUP: {
                EditTargetKind.PAGE,
                EditTargetKind.ARTIFACT,
            },
            EditDomain.RENDER_STYLE: {EditTargetKind.PARENT},
            EditDomain.RENDER_LAYOUT: {EditTargetKind.PARENT},
            EditDomain.REVIEW_METADATA: {EditTargetKind.PARENT},
            EditDomain.GLOSSARY: {EditTargetKind.PROJECT},
            EditDomain.LEDGER_CONTROL: {EditTargetKind.EDIT},
        }[domain]
        if target.kind not in allowed_targets:
            raise ValueError(
                f"{domain.value} edits cannot target {target.kind.value}"
            )
        if domain is EditDomain.STRUCTURAL:
            if operation == "set_reading_order":
                if target.kind is not EditTargetKind.PAGE:
                    raise ValueError(
                        "set_reading_order must target the complete page"
                    )
            elif target.kind is not EditTargetKind.PARENT:
                raise ValueError(
                    f"structural {operation} edits cannot target {target.kind.value}"
                )
            if operation == "add_user_parent":
                validate_user_parent_identity_pair(
                    target.parent_id,
                    str(frozen_payload.get("root_id") or ""),
                )
            elif operation == "split_user_parent":
                validate_user_parent_identity_pair(
                    target.parent_id,
                    str(frozen_payload.get("source_root_id") or ""),
                )
                if target.parent_id in set(
                    str(value)
                    for value in frozen_payload.get("child_parent_ids") or ()
                ):
                    raise ValueError(
                        "split child parent identities must differ from the source"
                    )
                if str(frozen_payload.get("source_root_id") or "") in set(
                    str(value)
                    for value in frozen_payload.get("child_root_ids") or ()
                ):
                    raise ValueError(
                        "split child root identities must differ from the source"
                    )
            elif operation == "merge_pipeline_parents":
                validate_user_parent_identity_pair(
                    target.parent_id,
                    str(frozen_payload.get("merged_root_id") or ""),
                )
                if target.parent_id in set(
                    str(value)
                    for value in frozen_payload.get("source_parent_ids") or ()
                ):
                    raise ValueError(
                        "merged parent identity must differ from source parents"
                    )
                if str(frozen_payload.get("merged_root_id") or "") in set(
                    str(value)
                    for value in frozen_payload.get("source_root_ids") or ()
                ):
                    raise ValueError(
                        "merged root identity must differ from source roots"
                    )
        if (
            domain is EditDomain.CLEANUP
            and target.kind is EditTargetKind.ARTIFACT
            and str(frozen_payload.get("revision_id") or "") != target.artifact_id
        ):
            raise ValueError(
                "cleanup selection target and payload revision IDs differ"
            )

    @property
    def is_control(self) -> bool:
        return self.domain is EditDomain.LEDGER_CONTROL

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_schema_version": self.edit_schema_version,
            "edit_id": self.edit_id,
            "project_id": self.project_id,
            "page_id": self.page_id,
            "target": self.target.to_dict(),
            "domain": self.domain.value,
            "operation": self.operation,
            "payload": thaw_json(self.payload),
            "base_revision_id": self.base_revision_id,
            "base_fingerprint": self.base_fingerprint,
            "supersedes_edit_id": self.supersedes_edit_id,
            "provenance": self.provenance,
            "created_at": self.created_at,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectEdit":
        """Deserialize a fresh record through the current strict contract."""

        return cls._from_dict(value, allow_unsupported_fill_color=False)

    @classmethod
    def from_persisted_dict(cls, value: Mapping[str, Any]) -> "ProjectEdit":
        """Load a persisted record while preserving legacy RGBA paint evidence.

        This compatibility entry point is reserved for trusted ledger/store
        readers.  Fresh command and public ``from_dict`` construction remain
        bound to the opaque ``#RRGGBB`` fill/outline write contract.
        """

        return cls._from_dict(value, allow_unsupported_fill_color=True)

    @classmethod
    def _from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        allow_unsupported_fill_color: bool,
    ) -> "ProjectEdit":
        if not isinstance(value, Mapping):
            raise TypeError("project edit must be a mapping")
        required = {
            "edit_schema_version",
            "edit_id",
            "project_id",
            "page_id",
            "target",
            "domain",
            "operation",
            "payload",
            "base_revision_id",
            "base_fingerprint",
            "supersedes_edit_id",
            "provenance",
            "created_at",
            "active",
        }
        missing = required - set(value)
        unknown = set(value) - required
        if missing:
            raise ValueError(f"project edit is missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"project edit has unsupported fields: {sorted(unknown)}")
        return cls(
            edit_schema_version=str(value.get("edit_schema_version") or ""),
            edit_id=str(value.get("edit_id") or ""),
            project_id=str(value.get("project_id") or ""),
            page_id=str(value.get("page_id") or ""),
            target=EditTarget.from_dict(value.get("target") or {}),
            domain=EditDomain(str(value.get("domain") or "")),
            operation=str(value.get("operation") or ""),
            payload=value.get("payload") or {},
            base_revision_id=str(value.get("base_revision_id") or ""),
            base_fingerprint=str(value.get("base_fingerprint") or ""),
            supersedes_edit_id=(
                str(value.get("supersedes_edit_id"))
                if value.get("supersedes_edit_id") is not None
                else None
            ),
            provenance=str(value.get("provenance") or ""),
            created_at=str(value.get("created_at") or ""),
            active=value.get("active"),
            _allow_unsupported_fill_color=allow_unsupported_fill_color,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_project_edit(
    *,
    project_id: str,
    page_id: str,
    target: EditTarget,
    domain: EditDomain,
    operation: str,
    payload: Mapping[str, Any],
    base_revision_id: str,
    base_fingerprint: str,
    supersedes_edit_id: str | None = None,
    provenance: str = "user",
    edit_id: str | None = None,
    created_at: str | None = None,
) -> ProjectEdit:
    return ProjectEdit(
        edit_id=str(edit_id or uuid.uuid4()),
        project_id=project_id,
        page_id=page_id,
        target=target,
        domain=domain,
        operation=operation,
        payload=payload,
        base_revision_id=base_revision_id,
        base_fingerprint=base_fingerprint,
        supersedes_edit_id=supersedes_edit_id,
        provenance=provenance,
        created_at=str(created_at or utc_now()),
        active=True,
    )
