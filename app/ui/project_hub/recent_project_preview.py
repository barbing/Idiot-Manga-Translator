"""Bounded, presentation-only discovery for recent-project thumbnails."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re


_MAX_PROJECT_PREFIX_CHARS = 256 * 1024
_JSON_STRING = r'"(?:[^"\\]|\\.)*"'
_PAGES = re.compile(r'"pages"\s*:\s*\[')
_ASSET_FIELDS = {
    field_name: re.compile(
        rf'"{field_name}"\s*:\s*(?P<value>{_JSON_STRING})'
    )
    for field_name in ("image_path", "output_path")
}


def recent_project_thumbnail_path(project_path: str) -> str:
    """Return one existing page image without loading project runtime state.

    Recent rows deliberately remain uninspected: this bounded prefix read does
    not migrate, validate, recover, or materialize a project.  It only resolves
    an existing source page (preferred) or rendered page for presentation.
    """

    project_file = Path(str(project_path or "").strip()).expanduser()
    try:
        project_file = project_file.resolve(strict=False)
        if not project_file.is_file():
            return ""
        with project_file.open("r", encoding="utf-8") as stream:
            prefix = stream.read(_MAX_PROJECT_PREFIX_CHARS)
    except (OSError, UnicodeError):
        return ""

    pages = _PAGES.search(prefix)
    if pages is None:
        return ""
    page_prefix = prefix[pages.end() :]
    for field_name in ("image_path", "output_path"):
        for match in _ASSET_FIELDS[field_name].finditer(page_prefix):
            try:
                raw_value = json.loads(match.group("value"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            candidate = Path(os.path.expandvars(raw_value)).expanduser()
            if not candidate.is_absolute():
                candidate = project_file.parent / candidate
            try:
                resolved = candidate.resolve(strict=False)
                if resolved.is_file():
                    return str(resolved)
            except OSError:
                continue
    return ""


__all__ = ["recent_project_thumbnail_path"]
