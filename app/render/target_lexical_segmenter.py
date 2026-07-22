# -*- coding: utf-8 -*-
"""Deterministic target-language lexical evidence for line breaking.

This module observes immutable spans in the unchanged translated text.  It
does not choose line breaks, rewrite text, mutate Jieba dictionaries, or make
layout decisions.  ``LineBreakPlanner`` remains the sole break selector.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Sequence

from app.render.typesetting_contracts import PunctuationToken, TextToken
from app.render.typesetting_text import strict_grapheme_clusters, tokens_original_text

_jieba_import_error: Exception | None = None
try:
    import jieba as _jieba
except Exception as exc:  # pragma: no cover - exercised by injected degradation
    _jieba = None
    _jieba_import_error = exc

_regex_import_error: Exception | None = None
try:
    import regex as _regex
except Exception as exc:  # pragma: no cover - declared runtime dependency
    _regex = None
    _regex_import_error = exc


TARGET_LEXICAL_SEGMENTER_VERSION = "target_lexical_segmenter_v1"
TARGET_LEXICAL_SPAN_VERSION = "target_lexical_span_v1"
_REQUIRED_JIEBA_VERSION = "0.42.1"
_HAN_CLUSTER_RE = _regex.compile(r"\A\p{Script=Han}+\Z") if _regex is not None else None


@dataclass(frozen=True)
class TargetLexicalSpan:
    """One immutable Jieba-backed lexical span in translated-text indices."""

    span_id: str
    text: str
    translated_start: int
    translated_end: int
    grapheme_start: int
    grapheme_end: int
    token_start: int
    token_end: int
    evidence_kind: str
    primary: bool

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "span_version": TARGET_LEXICAL_SPAN_VERSION,
            "span_id": self.span_id,
            "text": self.text,
            "translated_start": int(self.translated_start),
            "translated_end": int(self.translated_end),
            "grapheme_start": int(self.grapheme_start),
            "grapheme_end": int(self.grapheme_end),
            "token_start": int(self.token_start),
            "token_end": int(self.token_end),
            "evidence_kind": self.evidence_kind,
            "primary": bool(self.primary),
        }


@dataclass(frozen=True)
class TargetLexicalSegmentation:
    text: str
    spans: tuple[TargetLexicalSpan, ...] = ()
    available: bool = False
    status: str = "degraded"
    text_conserved: bool = False
    issues: tuple[str, ...] = ()
    package_version: str = ""
    dictionary_sha256: str = ""
    hmm_model_sha256: str = ""
    dictionary_path: str = ""
    hmm_model_files: tuple[str, ...] = ()

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "segmenter_version": TARGET_LEXICAL_SEGMENTER_VERSION,
            "status": self.status,
            "available": bool(self.available),
            "text_conserved": bool(self.text_conserved),
            "translated_text": self.text,
            "span_count": len(self.spans),
            "spans": [span.to_audit_dict() for span in self.spans],
            "issues": list(self.issues),
            "package": "jieba",
            "package_version": self.package_version,
            "required_package_version": _REQUIRED_JIEBA_VERSION,
            "mode": "accurate",
            "cut_all": False,
            "hmm": True,
            "tokenizer_scope": "dedicated_immutable_instance",
            "global_dictionary_mutation": False,
            "dictionary_sha256": self.dictionary_sha256,
            "hmm_model_sha256": self.hmm_model_sha256,
            "dictionary_path": self.dictionary_path,
            "hmm_model_files": list(self.hmm_model_files),
            "compound_evidence_policy": "same_dictionary_dag_overlapping_terms",
            "selection_authority": "LineBreakPlanner",
        }


@dataclass(frozen=True)
class TargetLexicalSegmenter:
    """Frozen wrapper around one pre-initialized, dedicated Jieba tokenizer."""

    _tokenizer: Any = field(default=None, repr=False, compare=False)
    package_version: str = ""
    dictionary_sha256: str = ""
    hmm_model_sha256: str = ""
    dictionary_path: str = ""
    hmm_model_files: tuple[str, ...] = ()
    available: bool = False
    issues: tuple[str, ...] = ()

    @classmethod
    def create(cls) -> "TargetLexicalSegmenter":
        if _jieba is None:
            return cls.unavailable(
                f"jieba_import_failed:{type(_jieba_import_error).__name__}"
            )
        if _HAN_CLUSTER_RE is None:
            return cls.unavailable(
                f"regex_import_failed:{type(_regex_import_error).__name__}"
            )
        try:
            package_version = importlib_metadata.version("jieba")
            package_root = Path(_jieba.__file__).resolve().parent
            dictionary = package_root / "dict.txt"
            hmm_files = tuple(
                package_root / "finalseg" / name
                for name in ("prob_start.py", "prob_trans.py", "prob_emit.py")
            )
            if package_version != _REQUIRED_JIEBA_VERSION:
                return cls.unavailable(
                    f"jieba_version_mismatch:{package_version}"
                )
            if not dictionary.is_file() or not all(path.is_file() for path in hmm_files):
                return cls.unavailable("jieba_model_asset_missing")
            tokenizer = _jieba.Tokenizer(dictionary=str(dictionary))
            tokenizer.initialize()
            return cls(
                _tokenizer=tokenizer,
                package_version=package_version,
                dictionary_sha256=_sha256_file(dictionary),
                hmm_model_sha256=_sha256_file_set(hmm_files, package_root),
                dictionary_path=str(dictionary),
                hmm_model_files=tuple(
                    path.relative_to(package_root).as_posix() for path in hmm_files
                ),
                available=True,
                issues=(),
            )
        except Exception as exc:  # pragma: no cover - environment-specific degradation
            return cls.unavailable(
                f"jieba_initialization_failed:{type(exc).__name__}"
            )

    @classmethod
    def unavailable(cls, reason: str) -> "TargetLexicalSegmenter":
        issue = str(reason or "jieba_unavailable")
        return cls(
            _tokenizer=None,
            available=False,
            issues=("target_lexical_segmenter_unavailable", issue),
        )

    def segment(
        self,
        text: str,
        tokens: Sequence[TextToken | PunctuationToken],
    ) -> TargetLexicalSegmentation:
        value = str(text or "")
        token_values = list(tokens or [])
        identity_conserved = tokens_original_text(token_values) == value
        if not self.available or self._tokenizer is None:
            return self._result(
                value,
                available=False,
                status="degraded",
                text_conserved=identity_conserved,
                issues=self.issues or ("target_lexical_segmenter_unavailable",),
            )
        if not identity_conserved:
            return self._result(
                value,
                available=False,
                status="degraded",
                text_conserved=False,
                issues=("target_lexical_token_identity_not_conserved",),
            )
        try:
            records = _han_chunk_records(value)
            raw_spans: list[tuple[int, int, str, bool]] = []
            for chunk_start, chunk_end, chunk in records:
                local_start = 0
                for word in self._tokenizer.cut(
                    chunk,
                    cut_all=False,
                    HMM=True,
                ):
                    local_end = local_start + len(word)
                    if chunk[local_start:local_end] != word:
                        raise RuntimeError(
                            "jieba_accurate_output_not_position_conserved"
                        )
                    raw_spans.append(
                        (
                            chunk_start + int(local_start),
                            chunk_start + int(local_end),
                            "jieba_accurate_hmm",
                            True,
                        )
                    )
                    local_start = local_end
                if local_start != len(chunk):
                    raise RuntimeError("jieba_accurate_output_not_conserved")
                dag = self._tokenizer.get_DAG(chunk)
                for local_start, inclusive_ends in sorted(dag.items()):
                    for inclusive_end in sorted(inclusive_ends):
                        if int(inclusive_end) <= int(local_start):
                            continue
                        raw_spans.append(
                            (
                                chunk_start + int(local_start),
                                chunk_start + int(inclusive_end) + 1,
                                "jieba_dictionary_dag",
                                False,
                            )
                        )
            spans = _materialize_spans(value, token_values, raw_spans)
            return self._result(
                value,
                spans=spans,
                available=True,
                status="ready",
                text_conserved=True,
                issues=(),
            )
        except Exception as exc:
            return self._result(
                value,
                available=False,
                status="degraded",
                text_conserved=identity_conserved,
                issues=(
                    "target_lexical_segmentation_failed",
                    f"target_lexical_segmentation_failed:{type(exc).__name__}",
                ),
            )

    def _result(
        self,
        text: str,
        *,
        spans: tuple[TargetLexicalSpan, ...] = (),
        available: bool,
        status: str,
        text_conserved: bool,
        issues: tuple[str, ...],
    ) -> TargetLexicalSegmentation:
        return TargetLexicalSegmentation(
            text=text,
            spans=spans,
            available=available,
            status=status,
            text_conserved=text_conserved,
            issues=tuple(_unique(issues)),
            package_version=self.package_version,
            dictionary_sha256=self.dictionary_sha256,
            hmm_model_sha256=self.hmm_model_sha256,
            dictionary_path=self.dictionary_path,
            hmm_model_files=self.hmm_model_files,
        )


@lru_cache(maxsize=1)
def default_target_lexical_segmenter() -> TargetLexicalSegmenter:
    """Return the one dedicated pre-initialized tokenizer for this process."""

    return TargetLexicalSegmenter.create()


def _han_chunk_records(text: str) -> list[tuple[int, int, str]]:
    clusters = strict_grapheme_clusters(text)
    records: list[tuple[int, int, str]] = []
    codepoint_offset = 0
    chunk_start: int | None = None
    chunk_parts: list[str] = []
    for cluster in clusters:
        start = codepoint_offset
        end = start + len(cluster)
        if _HAN_CLUSTER_RE is not None and _HAN_CLUSTER_RE.fullmatch(cluster):
            if chunk_start is None:
                chunk_start = start
            chunk_parts.append(cluster)
        elif chunk_start is not None:
            records.append((chunk_start, start, "".join(chunk_parts)))
            chunk_start = None
            chunk_parts = []
        codepoint_offset = end
    if chunk_start is not None:
        records.append((chunk_start, codepoint_offset, "".join(chunk_parts)))
    return records


def _materialize_spans(
    text: str,
    tokens: Sequence[TextToken | PunctuationToken],
    raw_spans: Sequence[tuple[int, int, str, bool]],
) -> tuple[TargetLexicalSpan, ...]:
    token_values = list(tokens)
    by_key: dict[tuple[int, int], tuple[str, bool]] = {}
    for start, end, evidence_kind, primary in raw_spans:
        if start < 0 or end <= start or end > len(text):
            continue
        prior = by_key.get((start, end))
        if prior is None or (primary and not prior[1]):
            by_key[(start, end)] = (evidence_kind, primary)

    spans: list[TargetLexicalSpan] = []
    for (start, end), (evidence_kind, primary) in sorted(
        by_key.items(),
        key=lambda item: (item[0][0], item[0][1], not item[1][1]),
    ):
        token_start = next(
            (
                index
                for index, token in enumerate(token_values)
                if int(token.translated_start) == int(start)
            ),
            None,
        )
        token_end_index = next(
            (
                index
                for index, token in enumerate(token_values)
                if int(token.translated_end) == int(end)
            ),
            None,
        )
        if token_start is None or token_end_index is None or token_end_index < token_start:
            continue
        selected = token_values[token_start : token_end_index + 1]
        value = text[start:end]
        if "".join(token.original_text for token in selected) != value:
            continue
        spans.append(
            TargetLexicalSpan(
                span_id=f"target_lexical_{len(spans):04d}",
                text=value,
                translated_start=int(start),
                translated_end=int(end),
                grapheme_start=int(selected[0].grapheme_start),
                grapheme_end=int(selected[-1].grapheme_end),
                token_start=int(token_start),
                token_end=int(token_end_index + 1),
                evidence_kind=evidence_kind,
                primary=bool(primary),
            )
        )
    return tuple(spans)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_set(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result
