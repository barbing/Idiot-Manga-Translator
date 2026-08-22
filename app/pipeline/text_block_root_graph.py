# -*- coding: utf-8 -*-
"""Root-owned text graph contract helpers.

This module is intentionally policy-only: it does not run detection, OCR,
translation, cleanup, or rendering. Controller code may collect evidence, but
root transaction status and graph audit semantics are centralized here.
"""
from __future__ import annotations

import json
import os
from typing import Any


ROOT_ACCEPTED = "root_accepted"
ROOT_PARTIAL_REVIEW = "root_partially_accepted_with_explicit_review_children"
ROOT_REVIEW_ONLY_UNRESOLVED = "root_review_only_unresolved"
ROOT_BLOCKED_NON_TEXT_OR_DECORATIVE = "root_blocked_non_text_or_decorative"
ROOT_CAPTION_BACKGROUND_REVIEW_ONLY = "root_caption_background_review_only"

ROOT_SPEECH = "speech_bubble"
ROOT_CAPTION = "caption_background"
ROOT_SFX = "sfx_decorative_art"
ROUTE_PRESERVE = "preserve"


_BLOCKING_PARENT_ACTIONS = {
    "source_quality_blocked",
    "block_auto_translation",
    "split_required",
    "unresolved_review",
    "block_review_only",
}

_LOW_QUALITY_REASON_CODES = {
    "low_japanese_source_ratio",
    "not_meaningful_caption_background_source",
    "caption_recovered_source_low_confidence",
    "speech_recovered_source_quality_blocked",
    "root_parent_source_rejected",
}



def parent_candidate_contract(
    root: Any,
    parent_candidate: dict[str, Any],
    *,
    accepted: bool,
    reasons: list[str] | None,
    score: float,
    child_status: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return the normalized audit contract for one root parent candidate."""
    child_candidates = [
        candidate
        for candidate in (parent_candidate.get("child_candidates") or [])
        if isinstance(candidate, dict)
    ]
    child_candidate_ids = [
        str(candidate.get("candidate_id") or "")
        for candidate in child_candidates
        if str(candidate.get("candidate_id") or "")
    ]
    included_child_region_ids = [
        str(candidate.get("source_region_id") or "")
        for candidate in child_candidates
        if str(candidate.get("source_region_id") or "")
    ]
    rejected_child_region_ids = [
        str(item.get("region_id") or "")
        for item in (child_status or [])
        if str(item.get("region_id") or "")
        and str(item.get("status") or "") in {
            "deliberately_rejected_child_fragment",
            "missing_meaningful_child_fragment",
        }
    ]
    missing_count = sum(
        1
        for item in (child_status or [])
        if str(item.get("status") or "") == "missing_meaningful_child_fragment"
    )
    record = dict(parent_candidate)
    record.update(
        {
            "parent_candidate_id": str(parent_candidate.get("parent_candidate_id") or ""),
            "root_id": str(getattr(root, "root_id", "") or parent_candidate.get("root_id") or ""),
            "root_type": str(getattr(root, "root_type", "") or parent_candidate.get("root_type") or ""),
            "source_text": str(parent_candidate.get("source_text") or ""),
            "evidence_scopes": sorted(
                {
                    str(candidate.get("source_scope") or "")
                    for candidate in child_candidates
                    if str(candidate.get("source_scope") or "")
                }
            ),
            "child_candidate_ids": child_candidate_ids,
            "included_child_region_ids": included_child_region_ids,
            "rejected_child_region_ids": rejected_child_region_ids,
            "source_quality_score": float(score or 0.0),
            "child_conservation_status": (
                "missing_meaningful_child"
                if missing_count
                else "complete"
            ),
            "accepted": bool(accepted),
            "rejection_reasons": [] if accepted else list(reasons or []),
            "acceptance_reasons": list(reasons or []) if accepted else [],
            "score": float(score or 0.0),
            "child_fragment_status": list(child_status or []),
        }
    )
    return record


def visual_parent_group_analysis(root: Any, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Group root-owned child candidates by visual continuity.

    This is deliberately geometry-only policy. It does not create ownership and it
    does not inspect page pixels. Controller-side reconstruction uses this to
    reject over-merged parent candidates before they can replace root children.
    """
    root_bbox = _bbox(getattr(root, "bbox", []) or [])
    records: list[dict[str, Any]] = []
    for candidate in candidates or []:
        bbox = _bbox(candidate.get("bbox") or [])
        body = _meaningful_body(candidate.get("ocr_text") or "")
        candidate_id = str(candidate.get("candidate_id") or "")
        if not bbox or not body or not candidate_id:
            continue
        records.append(
            {
                "candidate_id": candidate_id,
                "source_region_id": str(candidate.get("source_region_id") or ""),
                "bbox": bbox,
                "body": body,
                "ocr_text": str(candidate.get("ocr_text") or ""),
                "source_scope": str(candidate.get("source_scope") or ""),
            }
        )
    if not records:
        return {
            "status": "insufficient_geometry",
            "score": 0.0,
            "overmerge_risk": False,
            "rejection_reason": "",
            "groups": [],
            "candidate_group_map": {},
        }

    parent = {idx: idx for idx in range(len(records))}

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if _visual_boxes_connected(records[i], records[j], root_bbox):
                union(i, j)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)

    groups: list[dict[str, Any]] = []
    for group_index, items in enumerate(sorted(grouped.values(), key=_group_sort_key)):
        bbox = _union_bbox([item["bbox"] for item in items])
        group_id = f"vpg_{group_index:03d}"
        groups.append(
            {
                "parent_visual_group_id": group_id,
                "parent_visual_group_bbox": bbox,
                "parent_visual_group_child_ids": [item["candidate_id"] for item in items],
                "source_region_ids": [item["source_region_id"] for item in items if item["source_region_id"]],
                "source_scopes": sorted({item["source_scope"] for item in items if item["source_scope"]}),
                "source_text_preview": " / ".join(_short(item["ocr_text"], 24) for item in items),
            }
        )

    candidate_group_map = {
        candidate_id: group["parent_visual_group_id"]
        for group in groups
        for candidate_id in group.get("parent_visual_group_child_ids", [])
    }
    group_count = len(groups)
    container_count = len([cid for cid in (getattr(root, "text_area_container_ids", []) or []) if str(cid)])
    separated = group_count > 1
    multi_container_distinct = container_count > 1 and len(records) > 1 and _distinct_candidate_bodies(records)
    score = _visual_separation_score(groups, root_bbox)
    reason = ""
    if separated:
        reason = "separated_visual_parent_groups"
    elif multi_container_distinct:
        reason = "multi_container_root_combines_distinct_candidates"
    status = "separated_visual_groups" if reason else "single_visual_group"
    return {
        "status": status,
        "score": round(score, 4),
        "overmerge_risk": bool(reason),
        "rejection_reason": reason,
        "groups": groups,
        "candidate_group_map": candidate_group_map,
        "root_container_count": container_count,
    }


def annotate_parent_candidate_visual_group(
    parent_candidate: dict[str, Any],
    visual_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Stamp parent candidate visual group and over-merge audit fields."""
    child_ids = [str(item) for item in (parent_candidate.get("child_candidate_ids") or []) if str(item)]
    group_map = visual_analysis.get("candidate_group_map") or {}
    groups = visual_analysis.get("groups") or []
    group_by_id = {str(group.get("parent_visual_group_id") or ""): group for group in groups}
    group_ids = sorted({str(group_map.get(child_id) or "") for child_id in child_ids if str(group_map.get(child_id) or "")})
    if len(group_ids) == 1:
        group = group_by_id.get(group_ids[0], {})
        parent_candidate["parent_visual_group_id"] = group_ids[0]
        parent_candidate["parent_visual_group_bbox"] = list(group.get("parent_visual_group_bbox") or [])
        parent_candidate["parent_visual_group_child_ids"] = list(group.get("parent_visual_group_child_ids") or child_ids)
        parent_candidate["reconstruction_rejected_for_visual_overmerge"] = False
        return parent_candidate
    if len(group_ids) > 1:
        parent_candidate["parent_visual_group_id"] = "overmerged:" + "+".join(group_ids)
        parent_candidate["parent_visual_group_bbox"] = _union_bbox(
            [group_by_id.get(group_id, {}).get("parent_visual_group_bbox") or [] for group_id in group_ids]
        )
        parent_candidate["parent_visual_group_child_ids"] = list(child_ids)
        parent_candidate["reconstruction_rejected_for_visual_overmerge"] = True
        parent_candidate["root_overmerge_rejection_reason"] = (
            str(visual_analysis.get("rejection_reason") or "")
            or "parent_candidate_spans_multiple_visual_groups"
        )
        return parent_candidate
    parent_candidate["parent_visual_group_id"] = ""
    parent_candidate["parent_visual_group_bbox"] = []
    parent_candidate["parent_visual_group_child_ids"] = []
    parent_candidate["reconstruction_rejected_for_visual_overmerge"] = False
    return parent_candidate


def visual_group_records_for_audit(root: Any, visual_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in visual_analysis.get("groups") or []:
        rows.append(
            {
                "root_id": str(getattr(root, "root_id", "") or ""),
                "root_type": str(getattr(root, "root_type", "") or ""),
                "root_visual_separation_status": visual_analysis.get("status"),
                "root_visual_separation_score": visual_analysis.get("score"),
                "root_overmerge_risk": visual_analysis.get("overmerge_risk"),
                "root_overmerge_rejection_reason": visual_analysis.get("rejection_reason"),
                "parent_visual_group_id": group.get("parent_visual_group_id"),
                "parent_visual_group_bbox": group.get("parent_visual_group_bbox"),
                "parent_visual_group_child_ids": group.get("parent_visual_group_child_ids"),
                "source_region_ids": group.get("source_region_ids"),
                "source_scopes": group.get("source_scopes"),
                "source_text_preview": group.get("source_text_preview"),
            }
        )
    return rows


def apply_strict_root_transaction_contract(
    root: Any,
    root_parents: list[Any],
    root_children: list[Any],
) -> dict[str, Any]:
    """Mutate a root with strict transaction status and invariant fields."""
    record = root_acceptance_invariant_record(root, root_parents, root_children)
    accepted_parent_ids = list(record["accepted_parent_ids"])
    review_child_ids = list(record["review_child_ids"])
    has_accepted_parent = bool(accepted_parent_ids)
    root_type = str(getattr(root, "root_type", "") or "")
    route_policy = str(getattr(root, "route_policy", "") or "")

    if root_type == ROOT_SFX or route_policy == ROUTE_PRESERVE:
        status = ROOT_BLOCKED_NON_TEXT_OR_DECORATIVE
        reason = "blocked_preserve"
        blocker = False
    elif root_type == ROOT_CAPTION and not has_accepted_parent:
        status = ROOT_CAPTION_BACKGROUND_REVIEW_ONLY
        reason = (
            str(getattr(root, "root_source_coherence_failure_reason", "") or "")
            or "caption_background_root_review_only"
        )
        blocker = bool(record["unresolved_meaningful_child_count"])
    elif not has_accepted_parent:
        status = ROOT_REVIEW_ONLY_UNRESOLVED
        reason = (
            str(getattr(root, "root_source_coherence_failure_reason", "") or "")
            or "no_accepted_parent_unit"
        )
        blocker = bool(
            record["unresolved_meaningful_child_count"]
            or int(getattr(root, "root_child_count", 0) or 0)
            or bool(getattr(root, "root_reconstruction_attempted", False))
        )
    elif review_child_ids or record["unresolved_meaningful_child_count"]:
        status = ROOT_PARTIAL_REVIEW
        reason = (
            str(getattr(root, "root_source_coherence_failure_reason", "") or "")
            or "accepted_parent_with_explicit_review_children"
        )
        blocker = False
    else:
        status = ROOT_ACCEPTED
        reason = (
            str(getattr(root, "root_transaction_reason", "") or "")
            if str(getattr(root, "root_transaction_reason", "") or "") not in {"", "not_evaluated"}
            else "accepted_parent_unit"
        )
        blocker = False

    setattr(root, "root_transaction_status", status)
    setattr(root, "root_transaction_reason", reason)
    setattr(root, "root_validation_blocker", bool(blocker))
    if status == ROOT_BLOCKED_NON_TEXT_OR_DECORATIVE:
        setattr(root, "root_source_coherence_status", "blocked_preserve")
        setattr(root, "root_source_coherence_failure_reason", None)
    elif status == ROOT_CAPTION_BACKGROUND_REVIEW_ONLY:
        setattr(root, "root_source_coherence_status", "caption_background_review_only")
        setattr(root, "root_source_coherence_failure_reason", reason)
    elif status == ROOT_REVIEW_ONLY_UNRESOLVED:
        setattr(root, "root_source_coherence_status", "review_only_unresolved")
        setattr(root, "root_source_coherence_failure_reason", reason)

    setattr(root, "root_has_accepted_parent", has_accepted_parent)
    setattr(root, "root_accepted_parent_ids", accepted_parent_ids)
    setattr(root, "root_rejected_parent_count", int(record["rejected_parent_count"]))
    setattr(root, "root_low_quality_parent_count", int(record["low_quality_parent_count"]))
    setattr(root, "root_unresolved_meaningful_child_count", int(record["unresolved_meaningful_child_count"]))
    setattr(root, "root_review_child_ids", review_child_ids)
    setattr(root, "root_acceptance_blocker", bool(blocker))
    setattr(root, "root_acceptance_blocker_reason", reason if blocker else "")
    return record


def root_acceptance_invariant_record(
    root: Any,
    root_parents: list[Any],
    root_children: list[Any],
) -> dict[str, Any]:
    accepted_parent_ids = [
        str(getattr(parent, "parent_id", "") or "")
        for parent in root_parents
        if _parent_is_active_translation_unit(parent)
    ]
    accepted_parent_ids = [pid for pid in accepted_parent_ids if pid]
    rejected_parent_count = sum(1 for parent in root_parents if not _parent_is_active_translation_unit(parent))
    low_quality_parent_count = sum(1 for parent in root_parents if _parent_has_low_quality_reason(parent))
    low_quality_parent_count += sum(
        1
        for attempt in (getattr(root, "root_reconstruction_rejected_attempts", []) or [])
        if _attempt_has_low_quality_reason(attempt)
    )
    review_children = [
        str(getattr(child, "child_id", "") or getattr(child, "source_region_id", "") or "")
        for child in root_children
        if _child_is_review_state(child)
    ]
    unresolved_meaningful = [
        child
        for child in root_children
        if _child_is_review_state(child) and _meaningful_source_text(getattr(child, "ocr_text", "") or "")
    ]
    status = str(getattr(root, "root_transaction_status", "") or "")
    blocker_reason = str(getattr(root, "root_acceptance_blocker_reason", "") or "")
    return {
        "root_id": str(getattr(root, "root_id", "") or ""),
        "root_type": str(getattr(root, "root_type", "") or ""),
        "transaction_status": status,
        "has_accepted_parent": bool(accepted_parent_ids),
        "rejected_parent_count": rejected_parent_count,
        "low_quality_parent_count": low_quality_parent_count,
        "unresolved_meaningful_child_count": len(unresolved_meaningful),
        "accepted_parent_ids": accepted_parent_ids,
        "review_child_ids": [rid for rid in review_children if rid],
        "blocker": bool(getattr(root, "root_validation_blocker", False)),
        "blocker_reason": blocker_reason or str(getattr(root, "root_transaction_reason", "") or ""),
    }


def _bbox(value: Any) -> list[int]:
    try:
        vals = [int(round(float(v))) for v in list(value or [])[:4]]
    except Exception:
        return []
    if len(vals) < 4 or vals[2] <= 0 or vals[3] <= 0:
        return []
    return vals


def _meaningful_body(text: Any) -> str:
    return "".join(
        ch
        for ch in str(text or "")
        if not ch.isspace()
        and ch not in "・…...、。，,.!?！？ー-:：;；「」『』()（）[]【】"
    )


def _distinct_candidate_bodies(records: list[dict[str, Any]]) -> bool:
    _ = records
    return False


def _visual_boxes_connected(a: dict[str, Any], b: dict[str, Any], root_bbox: list[int]) -> bool:
    abox = list(a.get("bbox") or [])
    bbox = list(b.get("bbox") or [])
    if not abox or not bbox:
        return False
    aarea = max(1.0, float(abox[2]) * float(abox[3]))
    barea = max(1.0, float(bbox[2]) * float(bbox[3]))
    small_large_ratio = min(aarea, barea) / max(aarea, barea)
    if small_large_ratio < 0.12:
        return False

    ax, ay, aw, ah = [float(v) for v in abox]
    bx, by, bw, bh = [float(v) for v in bbox]
    overlap_x = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_y = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    x_overlap_ratio = overlap_x / max(1.0, min(aw, bw))
    y_overlap_ratio = overlap_y / max(1.0, min(ah, bh))
    x_gap = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
    y_gap = max(0.0, max(ay, by) - min(ay + ah, by + bh))
    root_w = float(root_bbox[2]) if len(root_bbox) >= 4 else max(aw, bw)
    root_h = float(root_bbox[3]) if len(root_bbox) >= 4 else max(ah, bh)
    y_threshold = max(18.0, min(root_h * 0.035, 42.0), min(ah, bh) * 0.35)
    x_threshold = max(18.0, min(root_w * 0.06, 44.0), min(aw, bw) * 0.55)
    if x_overlap_ratio >= 0.45 and y_gap <= y_threshold:
        return True
    if y_overlap_ratio >= 0.45 and x_gap <= x_threshold:
        return True
    return False


def _overlap_ratio(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / max(1.0, min(aw * ah, bw * bh))


def _inside_ratio(inner: list[int], outer: list[int]) -> float:
    if not inner or not outer:
        return 0.0
    ix, iy, iw, ih = [float(v) for v in inner]
    ox, oy, ow, oh = [float(v) for v in outer]
    inter_w = max(0.0, min(ix + iw, ox + ow) - max(ix, ox))
    inter_h = max(0.0, min(iy + ih, oy + oh) - max(iy, oy))
    return (inter_w * inter_h) / max(1.0, iw * ih)


def _union_bbox(boxes: list[Any]) -> list[int]:
    valid = [_bbox(box) for box in boxes]
    valid = [box for box in valid if box]
    if not valid:
        return []
    x1 = min(box[0] for box in valid)
    y1 = min(box[1] for box in valid)
    x2 = max(box[0] + box[2] for box in valid)
    y2 = max(box[1] + box[3] for box in valid)
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def _group_sort_key(items: list[dict[str, Any]]) -> tuple[float, float]:
    bbox = _union_bbox([item.get("bbox") for item in items])
    if not bbox:
        return (0.0, 0.0)
    # Japanese manga speech is commonly vertical; keep top-to-bottom groups
    # stable, with right-to-left as the secondary order.
    return (float(bbox[1]), -float(bbox[0] + bbox[2]))


def _visual_separation_score(groups: list[dict[str, Any]], root_bbox: list[int]) -> float:
    if len(groups) <= 1:
        return 0.0
    root_h = float(root_bbox[3]) if len(root_bbox) >= 4 and root_bbox[3] else 1.0
    root_w = float(root_bbox[2]) if len(root_bbox) >= 4 and root_bbox[2] else 1.0
    max_gap = 0.0
    boxes = [list(group.get("parent_visual_group_bbox") or []) for group in groups]
    boxes = [box for box in boxes if len(box) >= 4]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            ax, ay, aw, ah = [float(v) for v in a]
            bx, by, bw, bh = [float(v) for v in b]
            x_gap = max(0.0, max(ax, bx) - min(ax + aw, bx + bw)) / max(1.0, root_w)
            y_gap = max(0.0, max(ay, by) - min(ay + ah, by + bh)) / max(1.0, root_h)
            max_gap = max(max_gap, x_gap, y_gap)
    return float(len(groups) - 1) + max_gap


def write_root_graph_debug_artifacts(
    *,
    page_dir: str,
    hierarchy: dict[str, Any],
    root_reconstruction_executor: dict[str, Any] | None = None,
) -> dict[str, str]:
    os.makedirs(page_dir, exist_ok=True)
    executor = root_reconstruction_executor or {}
    paths: dict[str, str] = {}
    roots = hierarchy.get("text_area_root_blocks", []) or []
    parents = hierarchy.get("parent_logical_text_units", []) or []
    children = hierarchy.get("child_recognized_text_segments", []) or []
    parents_by_root: dict[str, list[dict[str, Any]]] = {}
    children_by_root: dict[str, list[dict[str, Any]]] = {}
    for parent in parents:
        parents_by_root.setdefault(str(parent.get("root_id") or ""), []).append(parent)
    for child in children:
        children_by_root.setdefault(str(child.get("root_id") or ""), []).append(child)

    invariant_rows = []
    transaction_rows = []
    caption_rows = []
    for root in roots:
        rid = str(root.get("root_id") or "")
        root_parents = parents_by_root.get(rid, [])
        root_children = children_by_root.get(rid, [])
        invariant = _dict_root_invariant(root, root_parents, root_children)
        invariant_rows.append(
            [
                invariant["root_id"],
                invariant["root_type"],
                root.get("root_transaction_status"),
                invariant["has_accepted_parent"],
                invariant["rejected_parent_count"],
                invariant["low_quality_parent_count"],
                invariant["unresolved_meaningful_child_count"],
                ",".join(invariant["accepted_parent_ids"]),
                ",".join(invariant["review_child_ids"]),
                root.get("root_validation_blocker"),
                root.get("root_acceptance_blocker_reason") or root.get("root_transaction_reason"),
            ]
        )
        transaction_rows.append(
            [
                rid,
                root.get("root_type"),
                root.get("route_policy"),
                root.get("root_transaction_status"),
                root.get("root_transaction_reason"),
                root.get("root_source_coherence_status"),
                root.get("root_reconstruction_status"),
                root.get("root_validation_blocker"),
                ",".join(root.get("root_accepted_parent_ids") or invariant["accepted_parent_ids"]),
            ]
        )
        if root.get("root_type") == ROOT_CAPTION:
            caption_rows.append(
                [
                    rid,
                    ",".join(root.get("text_area_container_ids") or []),
                    root.get("root_transaction_status"),
                    root.get("root_source_coherence_status"),
                    root.get("root_child_count"),
                    len(root_parents),
                    " / ".join(_short(parent.get("source_text"), 36) for parent in root_parents),
                    root.get("root_transaction_reason"),
                ]
            )

    paths["root_acceptance_invariant_table"] = _write_table(
        os.path.join(page_dir, "root_acceptance_invariant_table.md"),
        [
            "root_id",
            "root_type",
            "transaction_status",
            "has_accepted_parent",
            "rejected_parent_count",
            "low_quality_parent_count",
            "unresolved_meaningful_child_count",
            "accepted_parent_ids",
            "review_child_ids",
            "blocker",
            "blocker_reason",
        ],
        invariant_rows,
    )
    paths["root_transaction_table"] = _write_table(
        os.path.join(page_dir, "root_transaction_table.md"),
        [
            "root_id",
            "root_type",
            "route_policy",
            "transaction_status",
            "transaction_reason",
            "source_coherence",
            "reconstruction_status",
            "blocker",
            "accepted_parent_ids",
        ],
        transaction_rows,
    )
    paths["caption_background_root_recovery_table"] = _write_table(
        os.path.join(page_dir, "caption_background_root_recovery_table.md"),
        ["root_id", "containers", "transaction_status", "coherence", "children", "parents", "source", "reason"],
        caption_rows,
    )

    parent_rows = []
    child_rows = []
    full_page_rows = []
    overmerge_rows = []
    visual_group_rows = []
    reconstruction_rejection_rows = []
    before_after_rows = []
    for attempt in executor.get("attempts", []) or []:
        root_id = str(attempt.get("root_id") or "")
        ms = attempt.get("multi_scope_ctd_evidence") or {}
        visual = ms.get("visual_separation") or attempt.get("visual_separation") or {}
        if visual:
            overmerge_rows.append(
                [
                    root_id,
                    attempt.get("root_type"),
                    visual.get("status"),
                    visual.get("score"),
                    visual.get("overmerge_risk"),
                    visual.get("rejection_reason"),
                    len(visual.get("groups") or []),
                    attempt.get("status"),
                    _short(attempt.get("after_source"), 56),
                ]
            )
            for group in visual.get("groups") or []:
                visual_group_rows.append(
                    [
                        root_id,
                        group.get("parent_visual_group_id"),
                        group.get("parent_visual_group_bbox"),
                        ",".join(group.get("parent_visual_group_child_ids") or []),
                        ",".join(group.get("source_region_ids") or []),
                        ",".join(group.get("source_scopes") or []),
                        _short(group.get("source_text_preview"), 56),
                    ]
                )
        before_after_rows.append(
            [
                root_id,
                attempt.get("root_type"),
                " / ".join(_short(item, 32) for item in (attempt.get("before_sources") or [])),
                _short(attempt.get("after_source"), 56),
                attempt.get("status"),
                attempt.get("selected_variant"),
                attempt.get("new_block_id") or ",".join(str(item) for item in (attempt.get("new_block_ids") or [])),
            ]
        )
        for parent in ms.get("parent_candidates") or []:
            parent_rows.append(
                [
                    root_id,
                    parent.get("parent_candidate_id"),
                    parent.get("root_type"),
                    parent.get("accepted"),
                    _short(parent.get("source_text"), 56),
                    ",".join(parent.get("evidence_scopes") or []),
                    ",".join(parent.get("child_candidate_ids") or []),
                    ",".join(parent.get("included_child_region_ids") or []),
                    ",".join(parent.get("rejected_child_region_ids") or []),
                    parent.get("child_conservation_status"),
                    parent.get("source_quality_score"),
                    ",".join(parent.get("acceptance_reasons") or parent.get("rejection_reasons") or []),
                    parent.get("parent_visual_group_id"),
                    parent.get("parent_visual_group_bbox"),
                    ",".join(parent.get("parent_visual_group_child_ids") or []),
                    parent.get("reconstruction_rejected_for_visual_overmerge"),
                    parent.get("root_overmerge_rejection_reason"),
                ]
            )
            if parent.get("reconstruction_rejected_for_visual_overmerge"):
                reconstruction_rejection_rows.append(
                    [
                        root_id,
                        parent.get("parent_candidate_id"),
                        _short(parent.get("source_text"), 56),
                        parent.get("parent_visual_group_id"),
                        parent.get("parent_visual_group_bbox"),
                        parent.get("root_overmerge_rejection_reason"),
                        ",".join(parent.get("child_candidate_ids") or []),
                    ]
                )
        for record in ms.get("candidate_graph_records") or []:
            child_rows.append(
                [
                    root_id,
                    record.get("candidate_id"),
                    record.get("source_scope"),
                    record.get("source_region_id"),
                    _short(record.get("ocr_text"), 42),
                    record.get("root_overlap"),
                    record.get("center_in_root"),
                    record.get("role_compatible"),
                    record.get("sfx_decorative_conflict"),
                    record.get("candidate_graph_state"),
                    record.get("candidate_graph_reason"),
                    record.get("parent_candidate_id"),
                ]
            )
            if record.get("source_scope") == "full_page_ctd_evidence":
                full_page_rows.append(child_rows[-1])
        for candidate in ms.get("candidate_inventory") or []:
            if candidate.get("source_scope") != "full_page_ctd_evidence":
                continue
            if any(str(row[1]) == str(candidate.get("candidate_id")) and str(row[0]) == root_id for row in full_page_rows):
                continue
            full_page_rows.append(
                [
                    root_id,
                    candidate.get("candidate_id"),
                    candidate.get("source_scope"),
                    candidate.get("source_region_id"),
                    _short(candidate.get("ocr_text"), 42),
                    candidate.get("target_root_overlap_ratio"),
                    candidate.get("center_in_root"),
                    candidate.get("role_compatible"),
                    candidate.get("sfx_decorative_conflict"),
                    candidate.get("admission_status"),
                    ",".join(candidate.get("rejection_reasons") or []),
                    candidate.get("parent_candidate_id"),
                ]
            )

    paths["parent_candidate_acceptance_table"] = _write_table(
        os.path.join(page_dir, "parent_candidate_acceptance_table.md"),
        [
            "root_id",
            "parent_candidate_id",
            "root_type",
            "accepted",
            "source",
            "scopes",
            "child_candidate_ids",
            "included_child_region_ids",
            "rejected_child_region_ids",
            "child_conservation",
            "source_quality_score",
            "reasons",
            "parent_visual_group_id",
            "parent_visual_group_bbox",
            "parent_visual_group_child_ids",
            "reconstruction_rejected_for_visual_overmerge",
            "root_overmerge_rejection_reason",
        ],
        parent_rows,
    )
    paths["child_graph_state_table"] = _write_table(
        os.path.join(page_dir, "child_graph_state_table.md"),
        [
            "root_id",
            "candidate_id",
            "scope",
            "source_region_id",
            "ocr_text",
            "root_overlap",
            "center_in_root",
            "role_compatible",
            "sfx_conflict",
            "graph_state",
            "graph_reason",
            "parent_candidate_id",
        ],
        child_rows,
    )
    paths["full_page_ctd_admission_table"] = _write_table(
        os.path.join(page_dir, "full_page_ctd_admission_table.md"),
        [
            "root_id",
            "candidate_id",
            "scope",
            "source_region_id",
            "ocr_text",
            "root_overlap",
            "center_in_root",
            "role_compatible",
            "sfx_conflict",
            "state_or_status",
            "reason",
            "parent_candidate_id",
        ],
        full_page_rows,
    )
    paths["overmerged_root_table"] = _write_table(
        os.path.join(page_dir, "overmerged_root_table.md"),
        [
            "root_id",
            "root_type",
            "root_visual_separation_status",
            "root_visual_separation_score",
            "root_overmerge_risk",
            "root_overmerge_rejection_reason",
            "visual_group_count",
            "attempt_status",
            "after_source",
        ],
        overmerge_rows,
    )
    paths["visual_parent_group_table"] = _write_table(
        os.path.join(page_dir, "visual_parent_group_table.md"),
        [
            "root_id",
            "parent_visual_group_id",
            "parent_visual_group_bbox",
            "parent_visual_group_child_ids",
            "source_region_ids",
            "source_scopes",
            "source_text_preview",
        ],
        visual_group_rows,
    )
    paths["reconstruction_rejection_table"] = _write_table(
        os.path.join(page_dir, "reconstruction_rejection_table.md"),
        [
            "root_id",
            "parent_candidate_id",
            "source",
            "parent_visual_group_id",
            "parent_visual_group_bbox",
            "rejection_reason",
            "child_candidate_ids",
        ],
        reconstruction_rejection_rows,
    )
    paths["before_after_parent_unit_table"] = _write_table(
        os.path.join(page_dir, "before_after_parent_unit_table.md"),
        [
            "root_id",
            "root_type",
            "before_sources",
            "after_source",
            "attempt_status",
            "selected_variant",
            "new_block_id",
        ],
        before_after_rows,
    )
    return paths



def _parent_is_active_translation_unit(parent: Any) -> bool:
    if not bool(_get(parent, "translation_unit")):
        return False
    if str(_get(parent, "source_coherence_status") or "") == "rejected":
        return False
    if str(_get(parent, "source_coherence_action") or "") in _BLOCKING_PARENT_ACTIONS:
        return False
    source = _get(parent, "source_text") or ""
    if _meaningful_source_text(source):
        return True
    return (
        str(_get(parent, "role") or "") == "speech"
        and _short_speech_reaction_source(source)
    )


def _parent_has_low_quality_reason(parent: Any) -> bool:
    reasons = set(str(item) for item in (_get(parent, "reason_codes") or []))
    reasons.update(str(item) for item in (_get(parent, "source_coherence_reason_codes") or []))
    return bool(reasons & _LOW_QUALITY_REASON_CODES)


def _attempt_has_low_quality_reason(attempt: dict[str, Any]) -> bool:
    reasons = set(str(item) for item in (attempt.get("reasons") or []))
    return bool(reasons & _LOW_QUALITY_REASON_CODES)


def _child_is_review_state(child: Any) -> bool:
    state = str(_get(child, "final_state") or "")
    return state in {"unresolved_review_only", "blocked_by_root_policy", "noise_review_only"}


def _meaningful_source_text(text: Any) -> bool:
    body = "".join(ch for ch in str(text or "") if not ch.isspace() and ch not in "・…...、。，,.!?！？ー-:：;；「」『』()（）[]【】")
    return len(body) >= 2


def _short_speech_reaction_source(text: Any) -> bool:
    body = "".join(ch for ch in str(text or "") if not ch.isspace() and ch not in "・…...、。，,.!?！？ー-:：;；「」『』()（）[]【】")
    if len(body) != 1:
        return False
    return any(
        (0x3040 <= ord(ch) <= 0x30ff)
        or (0x3400 <= ord(ch) <= 0x9fff)
        for ch in body
    )


def _dict_root_invariant(root: dict[str, Any], parents: list[dict[str, Any]], children: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        str(parent.get("parent_id") or "")
        for parent in parents
        if _dict_parent_is_active_translation_unit(parent)
    ]
    review_children = [
        str(child.get("child_id") or child.get("source_region_id") or "")
        for child in children
        if str(child.get("final_state") or "") in {"unresolved_review_only", "blocked_by_root_policy", "noise_review_only"}
    ]
    unresolved = [
        child
        for child in children
        if str(child.get("final_state") or "") in {"unresolved_review_only", "blocked_by_root_policy", "noise_review_only"}
        and _meaningful_source_text(child.get("ocr_text"))
    ]
    low_quality = 0
    for parent in parents:
        reasons = set(str(item) for item in (parent.get("reason_codes") or []))
        reasons.update(str(item) for item in (parent.get("source_coherence_reason_codes") or []))
        if reasons & _LOW_QUALITY_REASON_CODES:
            low_quality += 1
    return {
        "root_id": str(root.get("root_id") or ""),
        "root_type": str(root.get("root_type") or ""),
        "has_accepted_parent": bool(accepted),
        "rejected_parent_count": sum(1 for parent in parents if str(parent.get("parent_id") or "") not in accepted),
        "low_quality_parent_count": low_quality,
        "unresolved_meaningful_child_count": len(unresolved),
        "accepted_parent_ids": accepted,
        "review_child_ids": [rid for rid in review_children if rid],
    }


def _dict_parent_is_active_translation_unit(parent: dict[str, Any]) -> bool:
    if not parent.get("translation_unit"):
        return False
    if parent.get("source_coherence_status") == "rejected":
        return False
    if parent.get("source_coherence_action") in _BLOCKING_PARENT_ACTIONS:
        return False
    if _meaningful_source_text(parent.get("source_text")):
        return True
    return (
        str(parent.get("role") or "") == "speech"
        and _short_speech_reaction_source(parent.get("source_text"))
    )


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _write_table(path: str, headers: list[str], rows: list[list[Any]]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(_md(value) for value in row) + " |\n")
    return path


def _short(value: Any, limit: int = 48) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "..."


def _md(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        text = ",".join(str(item) for item in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")
