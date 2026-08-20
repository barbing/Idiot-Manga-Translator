# -*- coding: utf-8 -*-
"""Typed contracts for one explicit, parent-scoped OCR source revision.

These values describe an application request and its immutable result.  They do
not select a detector, infer topology, expand a crop, or invoke a later pipeline
owner.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from app.config.settings_contracts import (
    RunSettingsSnapshot,
    canonical_fingerprint,
    freeze_json,
    run_settings_snapshot_from_dict,
    thaw_json,
)

from .hierarchy_revision_contracts import (
    EFFECTIVE_HIERARCHY_REVISION_PREFIX,
    validate_user_parent_identity_pair,
)


OCR_SOURCE_REVISION_SCHEMA_VERSION = "ocr_source_revision_v1"
OCR_SOURCE_REVISION_ID_PREFIX = "ocr-source-revision-v1-"
OCR_SOURCE_SELECTION_EDIT_ID_PREFIX = "ocr-source-selection-v1-"
SUPPORTED_OCR_ENGINES = frozenset({"PaddleOCR-VL", "MangaOCR"})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _require_identity(value: Any, field_name: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        raise ValueError(f"{field_name} is required")
    return candidate


def _require_path_safe_identity(value: Any, field_name: str) -> str:
    candidate = _require_identity(value, field_name)
    if _PATH_SAFE_ID.fullmatch(candidate) is None:
        raise ValueError(f"{field_name} must be path-safe")
    return candidate


def _require_sha256(value: Any, field_name: str) -> str:
    candidate = str(value or "").lower()
    if _SHA256.fullmatch(candidate) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return candidate


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive exact integer")
    return int(value)


def _require_bbox(
    value: Any,
    *,
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(
            "sampling_bbox must contain exact integer x, y, width, height"
        )
    x, y, width, height = (int(item) for item in value)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("sampling_bbox must be a positive page-bounded rectangle")
    page_width, page_height = canvas_size
    if x + width > page_width or y + height > page_height:
        raise ValueError("sampling_bbox must remain inside the original page")
    return x, y, width, height


class OcrRevisionErrorCode(str, Enum):
    CANCELLED = "cancelled"
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_LINEAGE_MISMATCH = "parent_lineage_mismatch"
    STALE_HIERARCHY = "stale_hierarchy"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    SOURCE_NOT_RUNNABLE = "source_not_runnable"
    SETTINGS_MISMATCH = "settings_mismatch"
    ORIGINAL_ASSET_UNAVAILABLE = "original_asset_unavailable"
    ORIGINAL_ASSET_MISMATCH = "original_asset_mismatch"
    RECOGNITION_FAILED = "recognition_failed"
    NON_AUTHORITATIVE_RESULT = "non_authoritative_result"
    EMPTY_RESULT = "empty_result"
    PROJECTION_REJECTED = "projection_rejected"
    PERSISTENCE_REJECTED = "persistence_rejected"


class OcrRevisionError(RuntimeError):
    """Typed fail-closed error for an explicit OCR revision transaction."""

    def __init__(self, code: OcrRevisionErrorCode, message: str) -> None:
        self.code = OcrRevisionErrorCode(code)
        super().__init__(str(message))


@dataclass(frozen=True)
class OriginalPageAssetBinding:
    asset_id: str
    asset_reference: str
    content_sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "asset_id",
            _require_identity(self.asset_id, "original_page.asset_id"),
        )
        object.__setattr__(
            self,
            "asset_reference",
            _require_identity(
                self.asset_reference,
                "original_page.asset_reference",
            ),
        )
        object.__setattr__(
            self,
            "content_sha256",
            _require_sha256(
                self.content_sha256,
                "original_page.content_sha256",
            ),
        )
        object.__setattr__(
            self,
            "width",
            _require_positive_int(self.width, "original_page.width"),
        )
        object.__setattr__(
            self,
            "height",
            _require_positive_int(self.height, "original_page.height"),
        )
        if self.width * self.height > 50_000_000:
            raise ValueError("original page exceeds the pixel safety limit")

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.width, self.height

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "asset_reference": self.asset_reference,
            "content_sha256": self.content_sha256,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OriginalPageAssetBinding":
        if not isinstance(value, Mapping):
            raise TypeError("original page binding must be a mapping")
        expected = {"asset_id", "asset_reference", "content_sha256", "width", "height"}
        if set(value) != expected:
            raise ValueError("original page binding fields are invalid")
        return cls(
            asset_id=value["asset_id"],
            asset_reference=value["asset_reference"],
            content_sha256=value["content_sha256"],
            width=value["width"],
            height=value["height"],
        )


@dataclass(frozen=True)
class ExplicitOcrRevisionRequest:
    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    expected_hierarchy_revision_id: str
    expected_hierarchy_fingerprint: str
    expected_effective_page_fingerprint: str
    original_page: OriginalPageAssetBinding
    sampling_bbox: tuple[int, int, int, int]
    run_settings_snapshot: RunSettingsSnapshot
    run_settings_fingerprint: str
    selected_ocr_engine: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_id",
            _require_path_safe_identity(self.command_id, "command_id"),
        )
        for field_name in (
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        if not str(self.expected_hierarchy_revision_id).startswith(
            EFFECTIVE_HIERARCHY_REVISION_PREFIX
        ):
            raise ValueError("expected_hierarchy_revision_id is invalid")
        for field_name in (
            "expected_hierarchy_fingerprint",
            "expected_effective_page_fingerprint",
            "run_settings_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        original = (
            self.original_page
            if isinstance(self.original_page, OriginalPageAssetBinding)
            else OriginalPageAssetBinding(**dict(self.original_page))
        )
        object.__setattr__(self, "original_page", original)
        object.__setattr__(
            self,
            "sampling_bbox",
            _require_bbox(self.sampling_bbox, canvas_size=original.canvas_size),
        )
        if not isinstance(self.run_settings_snapshot, RunSettingsSnapshot):
            raise TypeError("run_settings_snapshot must be a RunSettingsSnapshot")
        snapshot = self.run_settings_snapshot
        if snapshot.project_id != self.project_id:
            raise ValueError("run settings project identity does not match the request")
        if snapshot.unresolved_requirements:
            raise ValueError("run settings snapshot has unresolved requirements")
        if snapshot.settings_fingerprint != self.run_settings_fingerprint:
            raise ValueError("run settings fingerprint does not match the snapshot")
        selected = _require_identity(self.selected_ocr_engine, "selected_ocr_engine")
        if selected not in SUPPORTED_OCR_ENGINES:
            raise ValueError("selected_ocr_engine is unsupported")
        snapshot_selected = str(snapshot.pipeline_values.get("ocr_engine") or "")
        if snapshot_selected != selected:
            raise ValueError("selected OCR engine does not match the run snapshot")
        object.__setattr__(self, "selected_ocr_engine", selected)

    def to_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id,
            "project_id": self.project_id,
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "parent_authored_edit_id": self.parent_authored_edit_id,
            "expected_hierarchy_revision_id": self.expected_hierarchy_revision_id,
            "expected_hierarchy_fingerprint": self.expected_hierarchy_fingerprint,
            "expected_effective_page_fingerprint": (
                self.expected_effective_page_fingerprint
            ),
            "original_page": self.original_page.to_dict(),
            "sampling_bbox": list(self.sampling_bbox),
            "run_settings_snapshot": self.run_settings_snapshot.to_dict(),
            "run_settings_fingerprint": self.run_settings_fingerprint,
            "selected_ocr_engine": self.selected_ocr_engine,
            "expected_page_head_sha256": self.expected_page_head_sha256,
            "expected_global_head_sha256": self.expected_global_head_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplicitOcrRevisionRequest":
        if not isinstance(value, Mapping):
            raise TypeError("OCR revision request must be a mapping")
        expected = {
            "command_id",
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "expected_hierarchy_revision_id",
            "expected_hierarchy_fingerprint",
            "expected_effective_page_fingerprint",
            "original_page",
            "sampling_bbox",
            "run_settings_snapshot",
            "run_settings_fingerprint",
            "selected_ocr_engine",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        }
        if set(value) != expected:
            raise ValueError("OCR revision request fields are invalid")
        return cls(
            command_id=value["command_id"],
            project_id=value["project_id"],
            page_id=value["page_id"],
            parent_id=value["parent_id"],
            root_id=value["root_id"],
            parent_authored_edit_id=value["parent_authored_edit_id"],
            expected_hierarchy_revision_id=value[
                "expected_hierarchy_revision_id"
            ],
            expected_hierarchy_fingerprint=value[
                "expected_hierarchy_fingerprint"
            ],
            expected_effective_page_fingerprint=value[
                "expected_effective_page_fingerprint"
            ],
            original_page=OriginalPageAssetBinding.from_dict(
                value["original_page"]
            ),
            sampling_bbox=tuple(value["sampling_bbox"]),
            run_settings_snapshot=run_settings_snapshot_from_dict(
                value["run_settings_snapshot"]
            ),
            run_settings_fingerprint=value["run_settings_fingerprint"],
            selected_ocr_engine=value["selected_ocr_engine"],
            expected_page_head_sha256=value["expected_page_head_sha256"],
            expected_global_head_sha256=value["expected_global_head_sha256"],
        )


@dataclass(frozen=True)
class OcrRecognitionRequest:
    request: ExplicitOcrRevisionRequest
    crop: Any
    crop_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, ExplicitOcrRevisionRequest):
            raise TypeError("request must be an ExplicitOcrRevisionRequest")
        object.__setattr__(
            self,
            "crop_sha256",
            _require_sha256(self.crop_sha256, "crop_sha256"),
        )
        if self.crop is None:
            raise ValueError("crop is required")


@dataclass(frozen=True)
class OcrRecognitionReceipt:
    selected_ocr_engine: str
    text: str
    confidence: float
    authoritative: bool
    backend_name: str
    backend_metadata: Mapping[str, Any]
    recognition_metadata: Mapping[str, Any]
    crop_sha256: str

    def __post_init__(self) -> None:
        selected = _require_identity(self.selected_ocr_engine, "selected_ocr_engine")
        if selected not in SUPPORTED_OCR_ENGINES:
            raise ValueError("selected_ocr_engine is unsupported")
        object.__setattr__(self, "selected_ocr_engine", selected)
        if not isinstance(self.text, str):
            raise TypeError("recognized text must be a string")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
        ):
            raise ValueError("recognition confidence must be finite")
        object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.authoritative, bool):
            raise TypeError("authoritative must be a bool")
        object.__setattr__(
            self,
            "backend_name",
            _require_identity(self.backend_name, "backend_name"),
        )
        object.__setattr__(
            self,
            "backend_metadata",
            freeze_json(self.backend_metadata, field_name="backend_metadata"),
        )
        object.__setattr__(
            self,
            "recognition_metadata",
            freeze_json(
                self.recognition_metadata,
                field_name="recognition_metadata",
            ),
        )
        object.__setattr__(
            self,
            "crop_sha256",
            _require_sha256(self.crop_sha256, "crop_sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_ocr_engine": self.selected_ocr_engine,
            "text": self.text,
            "confidence": self.confidence,
            "authoritative": self.authoritative,
            "backend_name": self.backend_name,
            "backend_metadata": thaw_json(self.backend_metadata),
            "recognition_metadata": thaw_json(self.recognition_metadata),
            "crop_sha256": self.crop_sha256,
        }


@dataclass(frozen=True)
class OcrSourceRevisionArtifact:
    command_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    selection_edit_id: str
    source_text: str
    confidence: float
    original_page: OriginalPageAssetBinding
    sampling_bbox: tuple[int, int, int, int]
    crop_sha256: str
    run_settings_fingerprint: str
    selected_ocr_engine: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    input_effective_page_fingerprint: str
    backend_name: str
    backend_metadata: Mapping[str, Any]
    recognition_metadata: Mapping[str, Any]
    revision_id: str = ""
    provenance: str = "ocr_model_revision"
    schema_version: str = OCR_SOURCE_REVISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OCR_SOURCE_REVISION_SCHEMA_VERSION:
            raise ValueError("unsupported OCR source revision schema")
        if self.provenance != "ocr_model_revision":
            raise ValueError("OCR source revision provenance is invalid")
        for field_name in (
            "command_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "selection_edit_id",
            "selected_ocr_engine",
            "hierarchy_revision_id",
            "backend_name",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        if not isinstance(self.source_text, str) or not self.source_text.strip():
            raise ValueError("source revision text must be non-empty")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
        ):
            raise ValueError("source revision confidence must be finite")
        object.__setattr__(self, "confidence", float(self.confidence))
        original = (
            self.original_page
            if isinstance(self.original_page, OriginalPageAssetBinding)
            else OriginalPageAssetBinding(**dict(self.original_page))
        )
        object.__setattr__(self, "original_page", original)
        object.__setattr__(
            self,
            "sampling_bbox",
            _require_bbox(self.sampling_bbox, canvas_size=original.canvas_size),
        )
        for field_name in (
            "crop_sha256",
            "run_settings_fingerprint",
            "hierarchy_fingerprint",
            "input_effective_page_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        if self.selected_ocr_engine not in SUPPORTED_OCR_ENGINES:
            raise ValueError("selected OCR engine is unsupported")
        if not self.hierarchy_revision_id.startswith(
            EFFECTIVE_HIERARCHY_REVISION_PREFIX
        ):
            raise ValueError("hierarchy revision identity is invalid")
        for field_name in ("backend_metadata", "recognition_metadata"):
            object.__setattr__(
                self,
                field_name,
                freeze_json(getattr(self, field_name), field_name=field_name),
            )
        expected_revision_id = (
            OCR_SOURCE_REVISION_ID_PREFIX
            + canonical_fingerprint(self._semantic_dict())
        )
        if self.revision_id and self.revision_id != expected_revision_id:
            raise ValueError("source revision identity does not match its content")
        object.__setattr__(self, "revision_id", expected_revision_id)

    def _semantic_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance,
            "command_id": self.command_id,
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "parent_authored_edit_id": self.parent_authored_edit_id,
            "selection_edit_id": self.selection_edit_id,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "original_page": self.original_page.to_dict(),
            "sampling_bbox": list(self.sampling_bbox),
            "crop_sha256": self.crop_sha256,
            "run_settings_fingerprint": self.run_settings_fingerprint,
            "selected_ocr_engine": self.selected_ocr_engine,
            "hierarchy_revision_id": self.hierarchy_revision_id,
            "hierarchy_fingerprint": self.hierarchy_fingerprint,
            "input_effective_page_fingerprint": (
                self.input_effective_page_fingerprint
            ),
            "backend_name": self.backend_name,
            "backend_metadata": thaw_json(self.backend_metadata),
            "recognition_metadata": thaw_json(self.recognition_metadata),
        }

    def to_record(self, *, include_catalog: bool = False) -> dict[str, object]:
        result = self._semantic_dict()
        result["revision_id"] = self.revision_id
        if include_catalog:
            result["catalog"] = "source_revisions"
        return result

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "OcrSourceRevisionArtifact":
        if not isinstance(value, Mapping):
            raise TypeError("source revision artifact must be a mapping")
        allowed = {
            "schema_version",
            "provenance",
            "command_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "selection_edit_id",
            "source_text",
            "confidence",
            "original_page",
            "sampling_bbox",
            "crop_sha256",
            "run_settings_fingerprint",
            "selected_ocr_engine",
            "hierarchy_revision_id",
            "hierarchy_fingerprint",
            "input_effective_page_fingerprint",
            "backend_name",
            "backend_metadata",
            "recognition_metadata",
            "revision_id",
            "catalog",
        }
        unknown = frozenset(value) - allowed
        if unknown:
            raise ValueError(
                f"source revision artifact has unsupported fields: {sorted(unknown)}"
            )
        if "catalog" in value and value.get("catalog") != "source_revisions":
            raise ValueError("source revision artifact catalog is invalid")
        return cls(
            schema_version=str(value.get("schema_version") or ""),
            provenance=str(value.get("provenance") or ""),
            command_id=str(value.get("command_id") or ""),
            page_id=str(value.get("page_id") or ""),
            parent_id=str(value.get("parent_id") or ""),
            root_id=str(value.get("root_id") or ""),
            parent_authored_edit_id=str(
                value.get("parent_authored_edit_id") or ""
            ),
            selection_edit_id=str(value.get("selection_edit_id") or ""),
            source_text=value.get("source_text"),
            confidence=value.get("confidence"),
            original_page=OriginalPageAssetBinding(
                **dict(value.get("original_page") or {})
            ),
            sampling_bbox=tuple(value.get("sampling_bbox") or ()),
            crop_sha256=str(value.get("crop_sha256") or ""),
            run_settings_fingerprint=str(
                value.get("run_settings_fingerprint") or ""
            ),
            selected_ocr_engine=str(value.get("selected_ocr_engine") or ""),
            hierarchy_revision_id=str(value.get("hierarchy_revision_id") or ""),
            hierarchy_fingerprint=str(value.get("hierarchy_fingerprint") or ""),
            input_effective_page_fingerprint=str(
                value.get("input_effective_page_fingerprint") or ""
            ),
            backend_name=str(value.get("backend_name") or ""),
            backend_metadata=dict(value.get("backend_metadata") or {}),
            recognition_metadata=dict(value.get("recognition_metadata") or {}),
            revision_id=str(value.get("revision_id") or ""),
        )


@dataclass(frozen=True)
class ExplicitOcrRevisionReceipt:
    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    source_revision_id: str
    selection_edit_id: str
    source_text: str
    confidence: float
    selected_ocr_engine: str
    backend_name: str
    backend_metadata: Mapping[str, Any]
    recognition_metadata: Mapping[str, Any]
    original_page: OriginalPageAssetBinding
    sampling_bbox: tuple[int, int, int, int]
    crop_sha256: str
    run_settings_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: Mapping[str, Any]
    stage_requirements: tuple[Mapping[str, Any], ...]
    commit_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "command_id",
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "source_revision_id",
            "selection_edit_id",
            "selected_ocr_engine",
            "backend_name",
            "hierarchy_revision_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        if not isinstance(self.source_text, str) or not self.source_text.strip():
            raise ValueError("source_text must contain an authoritative OCR result")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
        ):
            raise ValueError("confidence must be finite")
        object.__setattr__(self, "confidence", float(self.confidence))
        for field_name in (
            "crop_sha256",
            "run_settings_fingerprint",
            "hierarchy_fingerprint",
            "before_effective_page_fingerprint",
            "after_effective_page_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        original = (
            self.original_page
            if isinstance(self.original_page, OriginalPageAssetBinding)
            else OriginalPageAssetBinding(**dict(self.original_page))
        )
        object.__setattr__(self, "original_page", original)
        object.__setattr__(
            self,
            "sampling_bbox",
            _require_bbox(self.sampling_bbox, canvas_size=original.canvas_size),
        )
        for field_name in (
            "backend_metadata",
            "recognition_metadata",
            "invalidation",
            "commit_receipt",
        ):
            object.__setattr__(
                self,
                field_name,
                freeze_json(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "stage_requirements",
            tuple(
                freeze_json(value, field_name="stage_requirements")
                for value in self.stage_requirements
            ),
        )
        OcrSourceRevisionArtifact(
            command_id=self.command_id,
            page_id=self.page_id,
            parent_id=self.parent_id,
            root_id=self.root_id,
            parent_authored_edit_id=self.parent_authored_edit_id,
            selection_edit_id=self.selection_edit_id,
            source_text=self.source_text,
            confidence=self.confidence,
            original_page=self.original_page,
            sampling_bbox=self.sampling_bbox,
            crop_sha256=self.crop_sha256,
            run_settings_fingerprint=self.run_settings_fingerprint,
            selected_ocr_engine=self.selected_ocr_engine,
            hierarchy_revision_id=self.hierarchy_revision_id,
            hierarchy_fingerprint=self.hierarchy_fingerprint,
            input_effective_page_fingerprint=(
                self.before_effective_page_fingerprint
            ),
            backend_name=self.backend_name,
            backend_metadata=self.backend_metadata,
            recognition_metadata=self.recognition_metadata,
            revision_id=self.source_revision_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id,
            "project_id": self.project_id,
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "parent_authored_edit_id": self.parent_authored_edit_id,
            "source_revision_id": self.source_revision_id,
            "selection_edit_id": self.selection_edit_id,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "selected_ocr_engine": self.selected_ocr_engine,
            "backend_name": self.backend_name,
            "backend_metadata": thaw_json(self.backend_metadata),
            "recognition_metadata": thaw_json(self.recognition_metadata),
            "original_page": self.original_page.to_dict(),
            "sampling_bbox": list(self.sampling_bbox),
            "crop_sha256": self.crop_sha256,
            "run_settings_fingerprint": self.run_settings_fingerprint,
            "hierarchy_revision_id": self.hierarchy_revision_id,
            "hierarchy_fingerprint": self.hierarchy_fingerprint,
            "before_effective_page_fingerprint": (
                self.before_effective_page_fingerprint
            ),
            "after_effective_page_fingerprint": self.after_effective_page_fingerprint,
            "invalidation": thaw_json(self.invalidation),
            "stage_requirements": [
                thaw_json(value) for value in self.stage_requirements
            ],
            "commit_receipt": thaw_json(self.commit_receipt),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplicitOcrRevisionReceipt":
        if not isinstance(value, Mapping):
            raise TypeError("OCR revision receipt must be a mapping")
        expected = {
            "command_id",
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "source_revision_id",
            "selection_edit_id",
            "source_text",
            "confidence",
            "selected_ocr_engine",
            "backend_name",
            "backend_metadata",
            "recognition_metadata",
            "original_page",
            "sampling_bbox",
            "crop_sha256",
            "run_settings_fingerprint",
            "hierarchy_revision_id",
            "hierarchy_fingerprint",
            "before_effective_page_fingerprint",
            "after_effective_page_fingerprint",
            "invalidation",
            "stage_requirements",
            "commit_receipt",
        }
        if set(value) != expected:
            raise ValueError("OCR revision receipt fields are invalid")
        return cls(
            command_id=value["command_id"],
            project_id=value["project_id"],
            page_id=value["page_id"],
            parent_id=value["parent_id"],
            root_id=value["root_id"],
            parent_authored_edit_id=value["parent_authored_edit_id"],
            source_revision_id=value["source_revision_id"],
            selection_edit_id=value["selection_edit_id"],
            source_text=value["source_text"],
            confidence=value["confidence"],
            selected_ocr_engine=value["selected_ocr_engine"],
            backend_name=value["backend_name"],
            backend_metadata=value["backend_metadata"],
            recognition_metadata=value["recognition_metadata"],
            original_page=OriginalPageAssetBinding.from_dict(
                value["original_page"]
            ),
            sampling_bbox=tuple(value["sampling_bbox"]),
            crop_sha256=value["crop_sha256"],
            run_settings_fingerprint=value["run_settings_fingerprint"],
            hierarchy_revision_id=value["hierarchy_revision_id"],
            hierarchy_fingerprint=value["hierarchy_fingerprint"],
            before_effective_page_fingerprint=value[
                "before_effective_page_fingerprint"
            ],
            after_effective_page_fingerprint=value[
                "after_effective_page_fingerprint"
            ],
            invalidation=value["invalidation"],
            stage_requirements=tuple(value["stage_requirements"]),
            commit_receipt=value["commit_receipt"],
        )


CancellationProbe = Callable[[], bool]


@runtime_checkable
class OcrRevisionRecognitionPort(Protocol):
    def recognize(
        self,
        request: OcrRecognitionRequest,
        *,
        cancellation_probe: CancellationProbe | None = None,
    ) -> OcrRecognitionReceipt:
        ...


@runtime_checkable
class ExplicitOcrRevisionPort(Protocol):
    def run_explicit_ocr_revision(
        self,
        request: ExplicitOcrRevisionRequest,
    ) -> ExplicitOcrRevisionReceipt:
        ...


__all__ = [
    "CancellationProbe",
    "ExplicitOcrRevisionPort",
    "ExplicitOcrRevisionReceipt",
    "ExplicitOcrRevisionRequest",
    "OCR_SOURCE_REVISION_ID_PREFIX",
    "OCR_SOURCE_REVISION_SCHEMA_VERSION",
    "OCR_SOURCE_SELECTION_EDIT_ID_PREFIX",
    "OcrRecognitionReceipt",
    "OcrRecognitionRequest",
    "OcrSourceRevisionArtifact",
    "OcrRevisionError",
    "OcrRevisionErrorCode",
    "OcrRevisionRecognitionPort",
    "OriginalPageAssetBinding",
    "SUPPORTED_OCR_ENGINES",
]
