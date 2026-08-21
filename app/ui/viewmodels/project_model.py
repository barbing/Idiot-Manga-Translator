# -*- coding: utf-8 -*-
"""Typed Qt model/view projections for projects, pages, and parents.

Rows are immutable values with stable identities.  The models never retain or
mutate a raw project dictionary.  PySide6 is optional at import time; when it
is available the public models are real ``QAbstractListModel`` subclasses.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, Iterable

from app.pipeline.hierarchy_revision_contracts import (
    ParentIdentityNamespace,
    ParentOrigin,
    ParentStageRequirement,
    RevisionRequiredAction,
    RevisionStage,
    RevisionStageState,
    RootIdentityNamespace,
)
from app.ui.ui_contract import Authority, Presentation
from app.ui.viewmodels.presentation_model import PagePresentationSnapshot


try:  # Import remains safe in contract-only/headless environments.
    from PySide6 import QtCore as _QtCore
except (ImportError, OSError):  # pragma: no cover - exercised without PySide.
    _QtCore = None


_USER_ROLE = 256


def _clean_required(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value.strip()


def _clean_optional(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value.strip()


def _require_count(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _enum_value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _presentation_values(presentation: Presentation) -> tuple[str, str, str]:
    if not isinstance(presentation, Presentation):
        raise TypeError("presentation must be Presentation")
    return (
        presentation.label,
        _enum_value(presentation.tone),
        presentation.icon,
    )


@dataclass(frozen=True, slots=True)
class ProjectRow:
    project_id: str
    name: str
    path: str
    language_pair: str
    page_count: int
    completed_count: int
    recoverable: bool
    presentation: Presentation
    thumbnail_path: str = ""
    accessibility_text: str = ""
    updated_label: str = ""

    def __post_init__(self) -> None:
        for field_name in ("project_id", "name", "path", "language_pair"):
            object.__setattr__(
                self,
                field_name,
                _clean_required(getattr(self, field_name), field_name),
            )
        page_count = _require_count(self.page_count, "page_count")
        completed_count = _require_count(self.completed_count, "completed_count")
        if completed_count > page_count:
            raise ValueError("completed_count cannot exceed page_count")
        _require_bool(self.recoverable, "recoverable")
        object.__setattr__(
            self,
            "thumbnail_path",
            _clean_optional(self.thumbnail_path, "thumbnail_path"),
        )
        object.__setattr__(
            self,
            "updated_label",
            _clean_optional(self.updated_label, "updated_label"),
        )
        status_label, _, _ = _presentation_values(self.presentation)
        accessibility = self.accessibility_text.strip()
        if not accessibility:
            accessibility = (
                f"{self.name}. {self.language_pair}. "
                f"{completed_count} of {page_count} pages complete. {status_label}."
            )
        object.__setattr__(self, "accessibility_text", accessibility)

    @property
    def stable_id(self) -> str:
        return self.project_id


@dataclass(frozen=True, slots=True)
class PageRow:
    page_id: str
    file_name: str
    ordinal: int
    parent_count: int
    progress_percent: int
    presentation: PagePresentationSnapshot
    thumbnail_path: str = ""
    elapsed_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "page_id", _clean_required(self.page_id, "page_id"))
        object.__setattr__(
            self,
            "file_name",
            _clean_required(self.file_name, "file_name"),
        )
        ordinal = _require_count(self.ordinal, "ordinal")
        if ordinal < 1:
            raise ValueError("ordinal must be at least one")
        _require_count(self.parent_count, "parent_count")
        progress = _require_count(self.progress_percent, "progress_percent")
        if progress > 100:
            raise ValueError("progress_percent cannot exceed 100")
        if not isinstance(self.presentation, PagePresentationSnapshot):
            raise TypeError("presentation must be PagePresentationSnapshot")
        object.__setattr__(
            self,
            "thumbnail_path",
            _clean_optional(self.thumbnail_path, "thumbnail_path"),
        )
        object.__setattr__(
            self,
            "elapsed_label",
            _clean_optional(self.elapsed_label, "elapsed_label"),
        )

    @property
    def stable_id(self) -> str:
        return self.page_id

    @property
    def accessibility_text(self) -> str:
        return f"{self.file_name}. {self.presentation.accessibility_text}"


@dataclass(frozen=True, slots=True)
class ParentRow:
    parent_id: str
    reading_order: int
    parent_role: str
    source_text: str | None
    target_text: str | None
    excluded: bool
    source_authority: Authority | None
    target_authority: Authority | None
    presentation: Presentation
    origin: ParentOrigin = ParentOrigin.AUTOMATIC
    identity_namespace: ParentIdentityNamespace = ParentIdentityNamespace.AUTOMATIC
    root_identity_namespace: RootIdentityNamespace = RootIdentityNamespace.AUTOMATIC
    stage_requirements: tuple[ParentStageRequirement, ...] = ()
    accessibility_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_id",
            _clean_required(self.parent_id, "parent_id"),
        )
        _require_count(self.reading_order, "reading_order")
        object.__setattr__(
            self,
            "parent_role",
            _clean_required(self.parent_role, "parent_role"),
        )
        for field_name in ("source_text", "target_text"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
        _require_bool(self.excluded, "excluded")
        for field_name in ("source_authority", "target_authority"):
            authority = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                None if authority is None else Authority(authority),
            )
        origin = ParentOrigin(self.origin)
        identity_namespace = ParentIdentityNamespace(self.identity_namespace)
        root_identity_namespace = RootIdentityNamespace(
            self.root_identity_namespace
        )
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "identity_namespace", identity_namespace)
        object.__setattr__(
            self,
            "root_identity_namespace",
            root_identity_namespace,
        )
        if not isinstance(self.stage_requirements, tuple) or any(
            not isinstance(item, ParentStageRequirement)
            for item in self.stage_requirements
        ):
            raise TypeError(
                "stage_requirements must contain ParentStageRequirement values"
            )
        if any(item.parent_id != self.parent_id for item in self.stage_requirements):
            raise ValueError("stage requirements belong to another parent")
        if origin is ParentOrigin.AUTOMATIC:
            if identity_namespace is not ParentIdentityNamespace.AUTOMATIC:
                raise ValueError("automatic parent requires the automatic namespace")
            if root_identity_namespace is not RootIdentityNamespace.AUTOMATIC:
                raise ValueError("automatic parent requires the automatic root namespace")
            if self.source_text is None or self.target_text is None:
                raise ValueError("automatic parent text must remain explicit strings")
            if self.source_authority is None or self.target_authority is None:
                raise ValueError("automatic parent authority must remain available")
        else:
            if identity_namespace is not ParentIdentityNamespace.USER_PARENT_V1:
                raise ValueError("user parent requires the user_parent_v1 namespace")
            if root_identity_namespace is not RootIdentityNamespace.USER_ROOT_V1:
                raise ValueError("user parent requires the user_root_v1 namespace")
            if not self.stage_requirements:
                raise ValueError("pending user parent requires typed stage requirements")
            if self.target_authority not in {None, Authority.USER_EDIT}:
                raise ValueError(
                    "user parent target authority cannot claim automatic provenance"
                )
            if self.target_authority is Authority.USER_EDIT and self.target_text is None:
                raise ValueError("user target authority requires exact target text")
            if self.source_authority not in {None, Authority.USER_EDIT}:
                raise ValueError(
                    "user parent source authority cannot claim automatic provenance"
                )
            requirements = {
                item.stage: item for item in self.stage_requirements
            }
            source = requirements.get(RevisionStage.SOURCE)
            translation = requirements.get(RevisionStage.TRANSLATION)
            if source is None or translation is None:
                raise ValueError(
                    "user parent requires source and translation stage facts"
                )
            if self.source_text is None:
                if self.source_authority is not None:
                    raise ValueError(
                        "unavailable user-parent source cannot claim edit authority"
                    )
                if not (
                    source.state is RevisionStageState.MISSING
                    and source.required_action
                    is RevisionRequiredAction.EXPLICIT_RUN
                ):
                    raise ValueError(
                        "missing user-parent source requires an explicit OCR run"
                    )
                if self.target_text is not None or self.target_authority is not None:
                    raise ValueError(
                        "source-unavailable user parent cannot expose target evidence"
                    )
            else:
                if not (
                    source.state is RevisionStageState.CURRENT
                    and source.required_action is RevisionRequiredAction.NONE
                ):
                    raise ValueError("selected user-parent source must be current")
                translation_missing = bool(
                    translation.state is RevisionStageState.MISSING
                    and translation.required_action
                    is RevisionRequiredAction.EXPLICIT_RUN
                )
                translation_current = bool(
                    translation.state is RevisionStageState.CURRENT
                    and translation.required_action is RevisionRequiredAction.NONE
                )
                if translation_missing:
                    if self.target_text is not None or self.target_authority is not None:
                        raise ValueError(
                            "translation-missing user parent cannot expose target evidence"
                        )
                elif translation_current:
                    if self.target_text is None:
                        raise ValueError(
                            "current user-parent translation requires exact target text"
                        )
                else:
                    raise ValueError(
                        "user-parent translation requirement is not a supported state"
                    )
        status_label, _, _ = _presentation_values(self.presentation)
        accessibility = self.accessibility_text.strip()
        if not accessibility:
            if origin is ParentOrigin.USER:
                accessibility = (
                    f"Parent {self.reading_order + 1}. {self.role_label}. "
                    f"User parent. {status_label}. Start and Preview unavailable "
                    "until required revisions are published."
                )
            else:
                accessibility = (
                    f"Parent {self.reading_order + 1}. {self.parent_role}. "
                    f"{status_label}."
                )
        object.__setattr__(self, "accessibility_text", accessibility)

    @property
    def stable_id(self) -> str:
        return self.parent_id

    @property
    def role_label(self) -> str:
        return {
            "speech": "Dialogue",
            "caption": "Caption",
        }.get(self.parent_role, self.parent_role.replace("_", " ").title())

    @property
    def execution_ready(self) -> bool:
        return all(
            item.state is RevisionStageState.CURRENT
            for item in self.stage_requirements
        )

    @property
    def display_text(self) -> str:
        if self.origin is ParentOrigin.USER:
            requirements = {
                item.stage: item for item in self.stage_requirements
            }
            source = requirements.get(RevisionStage.SOURCE)
            translation = requirements.get(RevisionStage.TRANSLATION)
            if (
                source is None
                or source.state is not RevisionStageState.CURRENT
            ):
                state = "Source required"
            elif (
                translation is None
                or translation.state is not RevisionStageState.CURRENT
            ):
                state = "Translation required"
            else:
                state = "Later revisions required"
            return f"{self.role_label} · User parent · {state}"
        return self.target_text or self.source_text or self.parent_id


class ProjectRole(IntEnum):
    STABLE_ID = _USER_ROLE + 1
    NAME = _USER_ROLE + 2
    PATH = _USER_ROLE + 3
    LANGUAGE_PAIR = _USER_ROLE + 4
    PAGE_COUNT = _USER_ROLE + 5
    COMPLETED_COUNT = _USER_ROLE + 6
    RECOVERABLE = _USER_ROLE + 7
    STATUS_LABEL = _USER_ROLE + 8
    STATUS_TONE = _USER_ROLE + 9
    STATUS_ICON = _USER_ROLE + 10
    ACCESSIBILITY_TEXT = _USER_ROLE + 11
    THUMBNAIL_PATH = _USER_ROLE + 12
    UPDATED_LABEL = _USER_ROLE + 13


class PageRole(IntEnum):
    STABLE_ID = _USER_ROLE + 1
    FILE_NAME = _USER_ROLE + 2
    ORDINAL = _USER_ROLE + 3
    OWNER = _USER_ROLE + 4
    PARENT_COUNT = _USER_ROLE + 5
    PROGRESS_PERCENT = _USER_ROLE + 6
    NEEDS_REVIEW = _USER_ROLE + 7
    THUMBNAIL_PATH = _USER_ROLE + 8
    STATUS_LABEL = _USER_ROLE + 9
    STATUS_TONE = _USER_ROLE + 10
    STATUS_ICON = _USER_ROLE + 11
    WORKSPACE_STATUS_LABEL = _USER_ROLE + 12
    WORKSPACE_STATUS_TONE = _USER_ROLE + 13
    WORKSPACE_STATUS_ICON = _USER_ROLE + 14
    ACCESSIBILITY_TEXT = _USER_ROLE + 15
    ELAPSED_LABEL = _USER_ROLE + 16


class ParentRole(IntEnum):
    STABLE_ID = _USER_ROLE + 1
    READING_ORDER = _USER_ROLE + 2
    PARENT_ROLE = _USER_ROLE + 3
    SOURCE_TEXT = _USER_ROLE + 4
    TARGET_TEXT = _USER_ROLE + 5
    EXCLUDED = _USER_ROLE + 6
    SOURCE_AUTHORITY = _USER_ROLE + 7
    TARGET_AUTHORITY = _USER_ROLE + 8
    STATUS_LABEL = _USER_ROLE + 9
    STATUS_TONE = _USER_ROLE + 10
    STATUS_ICON = _USER_ROLE + 11
    ACCESSIBILITY_TEXT = _USER_ROLE + 12
    ORIGIN = _USER_ROLE + 13
    IDENTITY_NAMESPACE = _USER_ROLE + 14
    ROOT_IDENTITY_NAMESPACE = _USER_ROLE + 15
    EXECUTION_READY = _USER_ROLE + 16


class _FallbackIndex:
    __slots__ = ("_row", "_column")

    def __init__(self, row: int = -1, column: int = -1) -> None:
        self._row = row
        self._column = column

    def row(self) -> int:
        return self._row

    def column(self) -> int:
        return self._column

    def isValid(self) -> bool:
        return self._row >= 0 and self._column >= 0


class _RowsMixin:
    _row_type: type[Any]
    _role_names: dict[int, bytes]
    _role_accessors: dict[int, Callable[[Any], Any]]

    @property
    def rows(self) -> tuple[Any, ...]:
        return self._rows

    @property
    def stable_ids(self) -> tuple[str, ...]:
        return tuple(row.stable_id for row in self._rows)

    def item_at(self, row: int) -> Any:
        if isinstance(row, bool) or not isinstance(row, int):
            raise TypeError("row must be an integer")
        return self._rows[row]

    def row_for_id(self, stable_id: str) -> int:
        value = _clean_required(stable_id, "stable_id")
        for row, item in enumerate(self._rows):
            if item.stable_id == value:
                return row
        return -1

    def _validated_rows(self, rows: Iterable[Any]) -> tuple[Any, ...]:
        normalized = tuple(rows)
        if any(not isinstance(row, self._row_type) for row in normalized):
            raise TypeError(f"rows must contain {self._row_type.__name__} values")
        stable_ids = tuple(row.stable_id for row in normalized)
        if len(stable_ids) != len(set(stable_ids)):
            raise ValueError("model rows must have unique stable IDs")
        return normalized

    def _data_for_row(self, row: Any, role: int) -> Any:
        accessor = self._role_accessors.get(int(role))
        return accessor(row) if accessor is not None else None


if _QtCore is not None:

    class TypedListModelBase(_RowsMixin, _QtCore.QAbstractListModel):
        def __init__(self, rows: Iterable[Any] = (), parent: Any = None) -> None:
            super().__init__(parent)
            self._rows: tuple[Any, ...] = ()
            self.replace_rows(rows)

        def rowCount(self, parent: Any = _QtCore.QModelIndex()) -> int:  # noqa: N802
            return 0 if parent.isValid() else len(self._rows)

        def data(self, index: Any, role: int = 0) -> Any:
            if not index.isValid() or not 0 <= index.row() < len(self._rows):
                return None
            row = self._rows[index.row()]
            if int(role) == int(_QtCore.Qt.ItemDataRole.DisplayRole):
                return self._display_value(row)
            if int(role) in {
                int(_QtCore.Qt.ItemDataRole.ToolTipRole),
                int(_QtCore.Qt.ItemDataRole.AccessibleTextRole),
            }:
                return row.accessibility_text
            return self._data_for_row(row, int(role))

        def roleNames(self) -> dict[int, bytes]:  # noqa: N802
            return dict(self._role_names)

        def replace_rows(self, rows: Iterable[Any]) -> None:
            normalized = self._validated_rows(rows)
            self.beginResetModel()
            self._rows = normalized
            self.endResetModel()

else:

    class TypedListModelBase(_RowsMixin):  # pragma: no cover - fallback only.
        def __init__(self, rows: Iterable[Any] = (), parent: Any = None) -> None:
            del parent
            self._rows: tuple[Any, ...] = ()
            self.replace_rows(rows)

        def index(self, row: int, column: int = 0, parent: Any = None) -> _FallbackIndex:
            del parent
            if 0 <= row < len(self._rows) and column == 0:
                return _FallbackIndex(row, column)
            return _FallbackIndex()

        def rowCount(self, parent: Any = None) -> int:  # noqa: N802
            if parent is not None and getattr(parent, "isValid", lambda: False)():
                return 0
            return len(self._rows)

        def data(self, index: Any, role: int = 0) -> Any:
            if not index.isValid() or not 0 <= index.row() < len(self._rows):
                return None
            row = self._rows[index.row()]
            if int(role) == 0:
                return self._display_value(row)
            return self._data_for_row(row, int(role))

        def roleNames(self) -> dict[int, bytes]:  # noqa: N802
            return dict(self._role_names)

        def replace_rows(self, rows: Iterable[Any]) -> None:
            self._rows = self._validated_rows(rows)


class ProjectListModel(TypedListModelBase):
    _row_type = ProjectRow
    _role_names = {
        ProjectRole.STABLE_ID: b"stableId",
        ProjectRole.NAME: b"name",
        ProjectRole.PATH: b"path",
        ProjectRole.LANGUAGE_PAIR: b"languagePair",
        ProjectRole.PAGE_COUNT: b"pageCount",
        ProjectRole.COMPLETED_COUNT: b"completedCount",
        ProjectRole.RECOVERABLE: b"recoverable",
        ProjectRole.STATUS_LABEL: b"statusLabel",
        ProjectRole.STATUS_TONE: b"statusTone",
        ProjectRole.STATUS_ICON: b"statusIcon",
        ProjectRole.ACCESSIBILITY_TEXT: b"accessibilityText",
        ProjectRole.THUMBNAIL_PATH: b"thumbnailPath",
        ProjectRole.UPDATED_LABEL: b"updatedLabel",
    }
    _role_accessors = {
        ProjectRole.STABLE_ID: lambda row: row.project_id,
        ProjectRole.NAME: lambda row: row.name,
        ProjectRole.PATH: lambda row: row.path,
        ProjectRole.LANGUAGE_PAIR: lambda row: row.language_pair,
        ProjectRole.PAGE_COUNT: lambda row: row.page_count,
        ProjectRole.COMPLETED_COUNT: lambda row: row.completed_count,
        ProjectRole.RECOVERABLE: lambda row: row.recoverable,
        ProjectRole.STATUS_LABEL: lambda row: row.presentation.label,
        ProjectRole.STATUS_TONE: lambda row: _enum_value(row.presentation.tone),
        ProjectRole.STATUS_ICON: lambda row: row.presentation.icon,
        ProjectRole.ACCESSIBILITY_TEXT: lambda row: row.accessibility_text,
        ProjectRole.THUMBNAIL_PATH: lambda row: row.thumbnail_path,
        ProjectRole.UPDATED_LABEL: lambda row: row.updated_label,
    }

    @staticmethod
    def _display_value(row: ProjectRow) -> str:
        return row.name


class PageListModel(TypedListModelBase):
    _row_type = PageRow
    _role_names = {
        PageRole.STABLE_ID: b"stableId",
        PageRole.FILE_NAME: b"fileName",
        PageRole.ORDINAL: b"ordinal",
        PageRole.OWNER: b"owner",
        PageRole.PARENT_COUNT: b"parentCount",
        PageRole.PROGRESS_PERCENT: b"progressPercent",
        PageRole.NEEDS_REVIEW: b"needsReview",
        PageRole.THUMBNAIL_PATH: b"thumbnailPath",
        PageRole.STATUS_LABEL: b"statusLabel",
        PageRole.STATUS_TONE: b"statusTone",
        PageRole.STATUS_ICON: b"statusIcon",
        PageRole.WORKSPACE_STATUS_LABEL: b"workspaceStatusLabel",
        PageRole.WORKSPACE_STATUS_TONE: b"workspaceStatusTone",
        PageRole.WORKSPACE_STATUS_ICON: b"workspaceStatusIcon",
        PageRole.ACCESSIBILITY_TEXT: b"accessibilityText",
        PageRole.ELAPSED_LABEL: b"elapsedLabel",
    }
    _role_accessors = {
        PageRole.STABLE_ID: lambda row: row.page_id,
        PageRole.FILE_NAME: lambda row: row.file_name,
        PageRole.ORDINAL: lambda row: row.ordinal,
        PageRole.OWNER: lambda row: row.presentation.workspace.owner,
        PageRole.PARENT_COUNT: lambda row: row.parent_count,
        PageRole.PROGRESS_PERCENT: lambda row: row.progress_percent,
        PageRole.NEEDS_REVIEW: lambda row: row.presentation.needs_review,
        PageRole.THUMBNAIL_PATH: lambda row: row.thumbnail_path,
        PageRole.STATUS_LABEL: lambda row: row.presentation.editor.label,
        PageRole.STATUS_TONE: lambda row: _enum_value(row.presentation.editor.tone),
        PageRole.STATUS_ICON: lambda row: row.presentation.editor.icon,
        PageRole.WORKSPACE_STATUS_LABEL: lambda row: row.presentation.workspace.label,
        PageRole.WORKSPACE_STATUS_TONE: lambda row: _enum_value(row.presentation.workspace.tone),
        PageRole.WORKSPACE_STATUS_ICON: lambda row: row.presentation.workspace.icon,
        PageRole.ACCESSIBILITY_TEXT: lambda row: row.accessibility_text,
        PageRole.ELAPSED_LABEL: lambda row: row.elapsed_label,
    }

    @staticmethod
    def _display_value(row: PageRow) -> str:
        return row.file_name


class ParentListModel(TypedListModelBase):
    _row_type = ParentRow
    _role_names = {
        ParentRole.STABLE_ID: b"stableId",
        ParentRole.READING_ORDER: b"readingOrder",
        ParentRole.PARENT_ROLE: b"parentRole",
        ParentRole.SOURCE_TEXT: b"sourceText",
        ParentRole.TARGET_TEXT: b"targetText",
        ParentRole.EXCLUDED: b"excluded",
        ParentRole.SOURCE_AUTHORITY: b"sourceAuthority",
        ParentRole.TARGET_AUTHORITY: b"targetAuthority",
        ParentRole.STATUS_LABEL: b"statusLabel",
        ParentRole.STATUS_TONE: b"statusTone",
        ParentRole.STATUS_ICON: b"statusIcon",
        ParentRole.ACCESSIBILITY_TEXT: b"accessibilityText",
        ParentRole.ORIGIN: b"origin",
        ParentRole.IDENTITY_NAMESPACE: b"identityNamespace",
        ParentRole.ROOT_IDENTITY_NAMESPACE: b"rootIdentityNamespace",
        ParentRole.EXECUTION_READY: b"executionReady",
    }
    _role_accessors = {
        ParentRole.STABLE_ID: lambda row: row.parent_id,
        ParentRole.READING_ORDER: lambda row: row.reading_order,
        ParentRole.PARENT_ROLE: lambda row: row.parent_role,
        ParentRole.SOURCE_TEXT: lambda row: row.source_text,
        ParentRole.TARGET_TEXT: lambda row: row.target_text,
        ParentRole.EXCLUDED: lambda row: row.excluded,
        ParentRole.SOURCE_AUTHORITY: lambda row: (
            row.source_authority.value
            if row.source_authority is not None
            else "unavailable"
        ),
        ParentRole.TARGET_AUTHORITY: lambda row: (
            row.target_authority.value
            if row.target_authority is not None
            else "unavailable"
        ),
        ParentRole.STATUS_LABEL: lambda row: row.presentation.label,
        ParentRole.STATUS_TONE: lambda row: _enum_value(row.presentation.tone),
        ParentRole.STATUS_ICON: lambda row: row.presentation.icon,
        ParentRole.ACCESSIBILITY_TEXT: lambda row: row.accessibility_text,
        ParentRole.ORIGIN: lambda row: row.origin.value,
        ParentRole.IDENTITY_NAMESPACE: lambda row: row.identity_namespace.value,
        ParentRole.ROOT_IDENTITY_NAMESPACE: lambda row: (
            row.root_identity_namespace.value
        ),
        ParentRole.EXECUTION_READY: lambda row: row.execution_ready,
    }

    @staticmethod
    def _display_value(row: ParentRow) -> str:
        return row.display_text


__all__ = [
    "PageListModel",
    "PageRole",
    "PageRow",
    "ParentListModel",
    "ParentRole",
    "ParentRow",
    "ProjectListModel",
    "ProjectRole",
    "ProjectRow",
    "TypedListModelBase",
]
