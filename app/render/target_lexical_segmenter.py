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
    import jieba.posseg as _jieba_pos
except Exception as exc:  # pragma: no cover - exercised by injected degradation
    _jieba = None
    _jieba_pos = None
    _jieba_import_error = exc

_regex_import_error: Exception | None = None
try:
    import regex as _regex
except Exception as exc:  # pragma: no cover - declared runtime dependency
    _regex = None
    _regex_import_error = exc


TARGET_LEXICAL_SEGMENTER_VERSION = "target_lexical_segmenter_v2_conflict_aware"
TARGET_LEXICAL_SPAN_VERSION = "target_lexical_span_v2"
TARGET_LEXICAL_BOUNDARY_VERSION = "target_lexical_boundary_v2"
_REQUIRED_JIEBA_VERSION = "0.42.1"
_HAN_CLUSTER_RE = _regex.compile(r"\A\p{Script=Han}+\Z") if _regex is not None else None
_MORPHOLOGICAL_ATTACHMENT_KINDS = frozenset(
    {
        "jieba_pos_suffix_attachment",
    }
)


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
    evidence_kinds: tuple[str, ...] = ()

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
            "evidence_kinds": list(self.evidence_kinds or (self.evidence_kind,)),
            "primary": bool(self.primary),
        }


@dataclass(frozen=True)
class TargetLexicalContribution:
    """One provider observation at a translated-text boundary.

    A contribution is evidence only.  It never makes a legal boundary
    illegal and never selects a layout.  ``evidence_family`` makes agreement
    and conflict explicit instead of treating overlapping Jieba output as
    additive truth.
    """

    contribution_id: str
    evidence_kind: str
    evidence_family: str
    relation: str
    rank: int
    text: str = ""
    translated_start: int = 0
    translated_end: int = 0
    details: tuple[tuple[str, str], ...] = ()

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "contribution_id": self.contribution_id,
            "evidence_kind": self.evidence_kind,
            "evidence_family": self.evidence_family,
            "relation": self.relation,
            "rank": int(self.rank),
            "text": self.text,
            "translated_start": int(self.translated_start),
            "translated_end": int(self.translated_end),
            "details": {key: value for key, value in self.details},
        }


@dataclass(frozen=True)
class TargetLexicalBoundary:
    """Conflict-aware evidence for one immutable token boundary."""

    boundary_id: str
    token_boundary: int
    translated_position: int
    grapheme_position: int
    state: str = "unknown"
    confirmed_keep_rank: int = 0
    weak_keep_rank: int = 0
    conflict: bool = False
    contributors: tuple[TargetLexicalContribution, ...] = ()

    @property
    def penalty(self) -> float:
        """Compatibility diagnostic; planner selection uses ordered ranks."""

        if self.confirmed_keep_rank > 0:
            return float(self.confirmed_keep_rank)
        if self.weak_keep_rank > 0:
            return float(self.weak_keep_rank) / 10.0
        return 0.0

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "boundary_version": TARGET_LEXICAL_BOUNDARY_VERSION,
            "boundary_id": self.boundary_id,
            "token_boundary": int(self.token_boundary),
            "translated_position": int(self.translated_position),
            "grapheme_position": int(self.grapheme_position),
            "state": self.state,
            "confirmed_keep_rank": int(self.confirmed_keep_rank),
            "weak_keep_rank": int(self.weak_keep_rank),
            "conflict": bool(self.conflict),
            "penalty": float(self.penalty),
            "contributors": [item.to_audit_dict() for item in self.contributors],
        }


@dataclass(frozen=True)
class _SpanObservation:
    start: int
    end: int
    evidence_kind: str
    evidence_family: str
    rank: int
    primary: bool = False
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _BoundaryObservation:
    position: int
    evidence_kind: str
    evidence_family: str
    relation: str
    rank: int
    start: int
    end: int
    text: str = ""
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TargetLexicalSegmentation:
    text: str
    spans: tuple[TargetLexicalSpan, ...] = ()
    boundaries: tuple[TargetLexicalBoundary, ...] = ()
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
            "boundary_count": len(self.boundaries),
            "boundaries": [item.to_audit_dict() for item in self.boundaries],
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
            "boundary_evidence_policy": (
                "ordered_hmm_nohmm_dag_pos_with_conflict_abstention"
            ),
            "absence_policy": "unknown_not_safe_break_truth",
            "conflict_policy": "abstain",
            "weak_evidence_scope": "same_size_same_topology_tiebreak_only",
            "morphological_attachment_scope": (
                "pre_topology_integrity_tier_conflict_abstaining"
            ),
            "selection_authority": "LineBreakPlanner",
        }


@dataclass(frozen=True)
class TargetLexicalSegmenter:
    """Frozen wrapper around one pre-initialized, dedicated Jieba tokenizer."""

    _tokenizer: Any = field(default=None, repr=False, compare=False)
    _pos_tokenizer: Any = field(default=None, repr=False, compare=False)
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
            pos_tokenizer = _jieba_pos.POSTokenizer(tokenizer)
            return cls(
                _tokenizer=tokenizer,
                _pos_tokenizer=pos_tokenizer,
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
            _pos_tokenizer=None,
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
            span_observations: list[_SpanObservation] = []
            boundary_observations: list[_BoundaryObservation] = []
            for chunk_start, chunk_end, chunk in records:
                hmm_spans, hmm_boundaries = _partition_observations(
                    self._tokenizer,
                    chunk,
                    chunk_start=chunk_start,
                    hmm=True,
                )
                no_hmm_spans, no_hmm_boundaries = _partition_observations(
                    self._tokenizer,
                    chunk,
                    chunk_start=chunk_start,
                    hmm=False,
                )
                span_observations.extend(hmm_spans)
                span_observations.extend(no_hmm_spans)
                boundary_observations.extend(hmm_boundaries)
                boundary_observations.extend(no_hmm_boundaries)
                span_observations.extend(
                    _dictionary_dag_observations(
                        self._tokenizer,
                        chunk,
                        chunk_start=chunk_start,
                    )
                )
                pos_spans, pos_boundaries = _pos_observations(
                    self._pos_tokenizer,
                    chunk,
                    chunk_start=chunk_start,
                )
                span_observations.extend(pos_spans)
                boundary_observations.extend(pos_boundaries)
            spans = _materialize_spans(
                value,
                token_values,
                span_observations,
            )
            boundaries = _materialize_boundaries(
                value,
                token_values,
                span_observations,
                boundary_observations,
            )
            return self._result(
                value,
                spans=spans,
                boundaries=boundaries,
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
        boundaries: tuple[TargetLexicalBoundary, ...] = (),
        available: bool,
        status: str,
        text_conserved: bool,
        issues: tuple[str, ...],
    ) -> TargetLexicalSegmentation:
        return TargetLexicalSegmentation(
            text=text,
            spans=spans,
            boundaries=boundaries,
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


def _partition_observations(
    tokenizer: Any,
    chunk: str,
    *,
    chunk_start: int,
    hmm: bool,
) -> tuple[list[_SpanObservation], list[_BoundaryObservation]]:
    evidence_kind = "jieba_accurate_hmm" if hmm else "jieba_accurate_no_hmm"
    evidence_family = "jieba_hmm_partition" if hmm else "jieba_no_hmm_partition"
    words = list(tokenizer.cut(chunk, cut_all=False, HMM=bool(hmm)))
    spans: list[_SpanObservation] = []
    boundaries: list[_BoundaryObservation] = []
    local_start = 0
    for index, word_value in enumerate(words):
        word = str(word_value)
        local_end = local_start + len(word)
        if chunk[local_start:local_end] != word:
            raise RuntimeError(f"{evidence_kind}_output_not_position_conserved")
        absolute_start = int(chunk_start + local_start)
        absolute_end = int(chunk_start + local_end)
        spans.append(
            _SpanObservation(
                start=absolute_start,
                end=absolute_end,
                evidence_kind=evidence_kind,
                evidence_family=evidence_family,
                rank=2,
                primary=bool(hmm),
                details=(("hmm", str(bool(hmm)).lower()),),
            )
        )
        if index < len(words) - 1:
            boundaries.append(
                _BoundaryObservation(
                    position=absolute_end,
                    evidence_kind=f"{evidence_family}_boundary",
                    evidence_family=evidence_family,
                    relation="break",
                    rank=1,
                    start=absolute_end,
                    end=absolute_end,
                    text="",
                    details=(("hmm", str(bool(hmm)).lower()),),
                )
            )
        local_start = local_end
    if local_start != len(chunk):
        raise RuntimeError(f"{evidence_kind}_output_not_conserved")
    return spans, boundaries


def _dictionary_dag_observations(
    tokenizer: Any,
    chunk: str,
    *,
    chunk_start: int,
) -> list[_SpanObservation]:
    observations: list[_SpanObservation] = []
    dag = tokenizer.get_DAG(chunk)
    for local_start, inclusive_ends in sorted(dag.items()):
        for inclusive_end in sorted(inclusive_ends):
            if int(inclusive_end) <= int(local_start):
                continue
            word = chunk[int(local_start) : int(inclusive_end) + 1]
            frequency = int(getattr(tokenizer, "FREQ", {}).get(word, 0) or 0)
            observations.append(
                _SpanObservation(
                    start=int(chunk_start + int(local_start)),
                    end=int(chunk_start + int(inclusive_end) + 1),
                    evidence_kind="jieba_dictionary_dag",
                    evidence_family="jieba_dictionary_dag",
                    rank=2,
                    primary=False,
                    details=(("dictionary_frequency", str(frequency)),),
                )
            )
    return observations


def _pos_observations(
    pos_tokenizer: Any,
    chunk: str,
    *,
    chunk_start: int,
) -> tuple[list[_SpanObservation], list[_BoundaryObservation]]:
    if pos_tokenizer is None:
        return [], []
    records = list(pos_tokenizer.cut(chunk, HMM=True))
    spans: list[_SpanObservation] = []
    boundaries: list[_BoundaryObservation] = []
    positioned: list[tuple[int, int, str, str]] = []
    local_start = 0
    for record in records:
        word = str(getattr(record, "word", ""))
        tag = str(getattr(record, "flag", ""))
        local_end = local_start + len(word)
        if chunk[local_start:local_end] != word:
            raise RuntimeError("jieba_pos_output_not_position_conserved")
        absolute_start = int(chunk_start + local_start)
        absolute_end = int(chunk_start + local_end)
        positioned.append((absolute_start, absolute_end, word, tag))
        spans.append(
            _SpanObservation(
                start=absolute_start,
                end=absolute_end,
                evidence_kind="jieba_pos_token",
                evidence_family="jieba_pos_token",
                rank=1,
                primary=False,
                details=(("pos", tag),),
            )
        )
        local_start = local_end
    if local_start != len(chunk):
        raise RuntimeError("jieba_pos_output_not_conserved")

    for left, right in zip(positioned, positioned[1:]):
        relation_kind = _pos_attachment_kind(left[2], left[3], right[2], right[3])
        if not relation_kind:
            continue
        boundaries.append(
            _BoundaryObservation(
                position=int(left[1]),
                evidence_kind=relation_kind,
                evidence_family="jieba_pos_attachment",
                relation="keep",
                rank=1,
                start=int(left[0]),
                end=int(right[1]),
                text=f"{left[2]}{right[2]}",
                details=(
                    ("left_pos", left[3]),
                    ("right_pos", right[3]),
                    ("left_text", left[2]),
                    ("right_text", right[2]),
                ),
            )
        )
    return spans, boundaries


def _pos_attachment_kind(
    left_word: str,
    left_tag: str,
    right_word: str,
    right_tag: str,
) -> str:
    """Return a grammatical or relational observation, never a word assertion."""

    left = str(left_tag or "").lower()
    right = str(right_tag or "").lower()
    # Jieba's ``u*`` tags describe postposed structural/aspect particles.
    # They attach to the lexical unit on their left; treating the boundary
    # after a particle as another keep relation incorrectly blocks natural
    # clause breaks such as ``...了|...``.
    if right.startswith("u"):
        return "jieba_pos_particle_attachment"
    if right.startswith("k") or str(right_word) == "们":
        return "jieba_pos_suffix_attachment"
    if right.startswith("f"):
        return "jieba_pos_locative_attachment"
    if left.startswith("n") and right.startswith("n"):
        return "jieba_pos_nominal_attachment"
    # Jieba tags compact adverbial numeral forms such as ``一起`` and ``一``
    # in ``一开始`` as ``m``.  When they directly modify a following verb,
    # their boundary is relational evidence for keeping the verbal unit
    # together.  This remains a weak POS observation, not a word assertion.
    if left.startswith("m") and right.startswith("v"):
        return "jieba_pos_adverbial_verb_attachment"
    # Jieba's structural subtypes ``uj`` (nominalizing 的), ``uv``
    # (adverbial 地), and ``uz`` (continuative 着) can bind a following
    # predicate.  Keep this narrower than generic ``u`` so clause markers
    # such as ``的话`` and sentence-final/aspect particles remain available
    # as natural break points.
    if left in {"uj", "uv", "uz"} and right.startswith("v"):
        return "jieba_pos_structural_predicate_attachment"
    # These relations are intentionally weak.  They describe ordinary
    # syntactic attachment in Jieba's emitted POS sequence, not a dictionary
    # word and not an illegal break.  The sole planner may use them only in
    # its late evidence tier when comparing otherwise equivalent layouts.
    if left.startswith("p") and right.startswith(("n", "r", "v")):
        return "jieba_pos_prepositional_attachment"
    if left.startswith("a") and right.startswith("n"):
        return "jieba_pos_attributive_attachment"
    if left.startswith("a") and right.startswith("v"):
        return "jieba_pos_modifier_predicate_attachment"
    if left.startswith("v") and right.startswith("n"):
        return "jieba_pos_predicate_nominal_attachment"
    if left.startswith("c") or right.startswith("c"):
        return "jieba_pos_coordination_attachment"
    return ""


def _materialize_spans(
    text: str,
    tokens: Sequence[TextToken | PunctuationToken],
    raw_spans: Sequence[_SpanObservation],
) -> tuple[TargetLexicalSpan, ...]:
    token_values = list(tokens)
    by_key: dict[tuple[int, int], list[_SpanObservation]] = {}
    for observation in raw_spans:
        start = int(observation.start)
        end = int(observation.end)
        if start < 0 or end <= start or end > len(text):
            continue
        by_key.setdefault((start, end), []).append(observation)

    spans: list[TargetLexicalSpan] = []
    for (start, end), observations in sorted(
        by_key.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        token_start, token_end_index = _token_range(token_values, start, end)
        if token_start is None or token_end_index is None or token_end_index < token_start:
            continue
        selected = token_values[token_start : token_end_index + 1]
        value = text[start:end]
        if "".join(token.original_text for token in selected) != value:
            continue
        ordered = sorted(
            observations,
            key=lambda item: (
                not bool(item.primary),
                -int(item.rank),
                item.evidence_kind,
            ),
        )
        primary = any(item.primary for item in ordered)
        evidence_kinds = tuple(_unique([item.evidence_kind for item in ordered]))
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
                evidence_kind=ordered[0].evidence_kind,
                primary=bool(primary),
                evidence_kinds=evidence_kinds,
            )
        )
    return tuple(spans)


def _materialize_boundaries(
    text: str,
    tokens: Sequence[TextToken | PunctuationToken],
    span_observations: Sequence[_SpanObservation],
    boundary_observations: Sequence[_BoundaryObservation],
) -> tuple[TargetLexicalBoundary, ...]:
    token_values = list(tokens)
    boundaries: list[TargetLexicalBoundary] = []
    for token_boundary in range(1, len(token_values)):
        after = token_values[token_boundary]
        position = int(after.translated_start)
        contributions: list[TargetLexicalContribution] = []
        for observation in span_observations:
            if not int(observation.start) < position < int(observation.end):
                continue
            contributions.append(
                _contribution(
                    len(contributions),
                    evidence_kind=observation.evidence_kind,
                    evidence_family=observation.evidence_family,
                    relation="keep",
                    rank=observation.rank,
                    text=text[int(observation.start) : int(observation.end)],
                    start=observation.start,
                    end=observation.end,
                    details=observation.details,
                )
            )
        for observation in boundary_observations:
            if int(observation.position) != position:
                continue
            contributions.append(
                _contribution(
                    len(contributions),
                    evidence_kind=observation.evidence_kind,
                    evidence_family=observation.evidence_family,
                    relation=observation.relation,
                    rank=observation.rank,
                    text=observation.text,
                    start=observation.start,
                    end=observation.end,
                    details=observation.details,
                )
            )
        contributions = _unique_contributions(contributions)
        families_by_relation = {
            relation: {
                item.evidence_family
                for item in contributions
                if item.relation == relation
            }
            for relation in ("keep", "break")
        }
        keep_families = families_by_relation["keep"]
        break_families = families_by_relation["break"]
        hmm_keep = "jieba_hmm_partition" in keep_families
        no_hmm_keep = "jieba_no_hmm_partition" in keep_families
        hmm_break = "jieba_hmm_partition" in break_families
        no_hmm_break = "jieba_no_hmm_partition" in break_families
        cross_partition_keep = bool(
            keep_families & {"jieba_dictionary_dag", "jieba_pos_token"}
        )
        conflict = bool(
            hmm_keep != no_hmm_keep
            or (cross_partition_keep and hmm_break and no_hmm_break)
        )
        confirmed_rank = 2 if hmm_keep and no_hmm_keep and not conflict else 0
        morphological_contributors = [
            item
            for item in contributions
            if item.relation == "keep"
            and item.evidence_kind in _MORPHOLOGICAL_ATTACHMENT_KINDS
        ]
        if not conflict and not confirmed_rank and morphological_contributors:
            confirmed_rank = max(
                int(item.rank) for item in morphological_contributors
            )
        weak_contributors = [
            item
            for item in contributions
            if item.relation == "keep"
            and item.evidence_family == "jieba_pos_attachment"
            and item.evidence_kind not in _MORPHOLOGICAL_ATTACHMENT_KINDS
        ]
        weak_rank = (
            max((int(item.rank) for item in weak_contributors), default=0)
            if not conflict and not confirmed_rank
            else 0
        )
        state = (
            "conflicted"
            if conflict
            else "confirmed_keep"
            if confirmed_rank
            else "weak_keep"
            if weak_rank
            else "unknown"
        )
        boundaries.append(
            TargetLexicalBoundary(
                boundary_id=f"target_lexical_boundary_{token_boundary:04d}",
                token_boundary=int(token_boundary),
                translated_position=position,
                grapheme_position=int(after.grapheme_start),
                state=state,
                confirmed_keep_rank=int(confirmed_rank),
                weak_keep_rank=int(weak_rank),
                conflict=conflict,
                contributors=tuple(contributions),
            )
        )
    return tuple(boundaries)


def _contribution(
    ordinal: int,
    *,
    evidence_kind: str,
    evidence_family: str,
    relation: str,
    rank: int,
    text: str,
    start: int,
    end: int,
    details: tuple[tuple[str, str], ...],
) -> TargetLexicalContribution:
    return TargetLexicalContribution(
        contribution_id=f"lexical_contribution_{ordinal:03d}",
        evidence_kind=str(evidence_kind),
        evidence_family=str(evidence_family),
        relation=str(relation),
        rank=max(0, int(rank)),
        text=str(text or ""),
        translated_start=int(start),
        translated_end=int(end),
        details=tuple((str(key), str(value)) for key, value in details),
    )


def _unique_contributions(
    values: Sequence[TargetLexicalContribution],
) -> list[TargetLexicalContribution]:
    result: list[TargetLexicalContribution] = []
    seen: set[tuple[Any, ...]] = set()
    for item in values:
        key = (
            item.evidence_kind,
            item.evidence_family,
            item.relation,
            item.rank,
            item.text,
            item.translated_start,
            item.translated_end,
            item.details,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(
            TargetLexicalContribution(
                contribution_id=f"lexical_contribution_{len(result):03d}",
                evidence_kind=item.evidence_kind,
                evidence_family=item.evidence_family,
                relation=item.relation,
                rank=item.rank,
                text=item.text,
                translated_start=item.translated_start,
                translated_end=item.translated_end,
                details=item.details,
            )
        )
    return result


def _token_range(
    tokens: Sequence[TextToken | PunctuationToken],
    start: int,
    end: int,
) -> tuple[int | None, int | None]:
    token_start = next(
        (
            index
            for index, token in enumerate(tokens)
            if int(token.translated_start) == int(start)
        ),
        None,
    )
    token_end = next(
        (
            index
            for index, token in enumerate(tokens)
            if int(token.translated_end) == int(end)
        ),
        None,
    )
    return token_start, token_end


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
