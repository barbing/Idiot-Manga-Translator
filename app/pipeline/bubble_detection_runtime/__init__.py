"""Production-owned model-evidence fusion runtime for BubbleDetection.

The runtime preserves the accepted Phase 4 provider contract while omitting
the former standalone diagnostic CLI and project-output dependencies.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


from . import kitsumed as kit
from . import ogkalu as og


def bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _overlap_ratio(inner: list[float], outer: list[float]) -> float:
    area = bbox_area(inner)
    return _intersection_area(inner, outer) / area if area > 0.0 else 0.0


def _bbox_center(box: list[float]) -> tuple[int, int]:
    return (
        int(round((box[0] + box[2]) / 2.0)),
        int(round((box[1] + box[3]) / 2.0)),
    )


def _center_inside_bbox(inner: list[float], outer: list[float]) -> bool:
    cx, cy = _bbox_center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def mask_overlap_ratio(box: list[float], mask: np.ndarray) -> float:
    height, width = mask.shape[:2]
    x1 = max(0, min(width, int(np.floor(box[0]))))
    y1 = max(0, min(height, int(np.floor(box[1]))))
    x2 = max(0, min(width, int(np.ceil(box[2]))))
    y2 = max(0, min(height, int(np.ceil(box[3]))))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = mask[y1:y2, x1:x2]
    return float(crop.sum()) / float(crop.size) if crop.size else 0.0


def center_inside_mask(box: list[float], mask: np.ndarray) -> bool:
    cx, cy = _bbox_center(box)
    height, width = mask.shape[:2]
    if cx < 0 or cy < 0 or cx >= width or cy >= height:
        return False
    return bool(mask[cy, cx])


def _region_bbox(region: dict[str, Any]) -> list[float] | None:
    bbox = region.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x, y, width, height = [float(value) for value in bbox]
    return [x, y, x + width, y + height]


def _is_decorative_or_sfx(region: dict[str, Any]) -> bool:
    semantic = str(region.get("semantic_class") or "")
    reason = str(region.get("classification_reason") or "").lower()
    return bool(
        semantic in {"decorative_text", "sfx"}
        or region.get("is_decorative")
        or region.get("is_sfx")
        or "sfx" in reason
        or "decorative" in reason
        or "art_text" in reason
        or "preserve" in str(region.get("skip_reason") or "").lower()
    )


def _is_caption_or_background(region: dict[str, Any]) -> bool:
    semantic = str(region.get("semantic_class") or "")
    reason = str(region.get("classification_reason") or "").lower()
    return bool(
        semantic in {"background_text", "sign_text"}
        or "caption" in reason
        or "background" in reason
    )


def _is_speech(region: dict[str, Any]) -> bool:
    return str(region.get("semantic_class") or "") == "speech_bubble" or bool(
        region.get("is_speech_bubble")
    )


def contour_polygon(mask: np.ndarray) -> list[list[int]]:
    binary = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(2.0, 0.01 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) > 80:
        step = max(1, len(approx) // 80)
        approx = approx[::step][:80]
    return [[int(x), int(y)] for x, y in approx]


def link_regions(
    regions: list[dict[str, Any]],
    kitsumed: list[dict[str, Any]],
    ogkalu: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for region in regions:
        region_id = str(region.get("region_id") or "")
        bbox = _region_bbox(region)
        if not region_id or bbox is None:
            continue
        kitsumed_links = []
        for detection in kitsumed:
            ratio = mask_overlap_ratio(bbox, detection["mask"])
            center_hit = center_inside_mask(bbox, detection["mask"])
            if ratio >= 0.05 or center_hit:
                kitsumed_links.append(
                    {
                        "evidence_id": detection["model_evidence_id"],
                        "mask_overlap_ratio": round(ratio, 4),
                        "center_inside_mask": center_hit,
                        "confidence": detection["confidence"],
                    }
                )
        ogkalu_links = []
        for detection in ogkalu:
            detection_box = [float(value) for value in detection["bbox_xyxy"]]
            ratio = _overlap_ratio(bbox, detection_box)
            center_hit = _center_inside_bbox(bbox, detection_box)
            if ratio >= 0.05 or center_hit:
                ogkalu_links.append(
                    {
                        "evidence_id": detection["model_evidence_id"],
                        "class_name": detection["class_name"],
                        "bbox_overlap_ratio": round(ratio, 4),
                        "center_inside_bbox": center_hit,
                        "confidence": detection["confidence"],
                    }
                )
        links.append(
            {
                "region_id": region_id,
                "bbox": [round(value, 2) for value in bbox],
                "ocr_text": region.get("ocr_text") or "",
                "semantic_class": region.get("semantic_class"),
                "cleanup_mode": region.get("cleanup_mode"),
                "classification_reason": region.get("classification_reason"),
                "diagnostic_text_container_id": region.get(
                    "diagnostic_text_container_id"
                ),
                "diagnostic_container_type": region.get(
                    "diagnostic_container_type"
                ),
                "diagnostic_route_suggestions": region.get(
                    "diagnostic_route_suggestions"
                ),
                "diagnostic_render_plan_suggestions": region.get(
                    "diagnostic_render_plan_suggestions"
                ),
                "is_decorative_or_sfx": _is_decorative_or_sfx(region),
                "is_caption_or_background": _is_caption_or_background(region),
                "is_speech": _is_speech(region),
                "kitsumed_links": kitsumed_links,
                "ogkalu_links": ogkalu_links,
            }
        )
    return links


def build_fusion(
    page: str,
    kitsumed: list[dict[str, Any]],
    ogkalu: list[dict[str, Any]],
    region_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    ogkalu_claimed: set[str] = set()
    for index, detection in enumerate(kitsumed):
        evidence_id = detection["model_evidence_id"]
        linked_ogkalu: list[str] = []
        has_text_bubble = False
        has_bubble = False
        has_text_free = False
        for text_detection in ogkalu:
            text_id = text_detection["model_evidence_id"]
            text_box = [float(value) for value in text_detection["bbox_xyxy"]]
            ratio = mask_overlap_ratio(text_box, detection["mask"])
            center_hit = center_inside_mask(text_box, detection["mask"])
            if ratio >= 0.08 or center_hit:
                linked_ogkalu.append(text_id)
                ogkalu_claimed.add(text_id)
                has_text_bubble = has_text_bubble or text_detection["class_name"] == "text_bubble"
                has_bubble = has_bubble or text_detection["class_name"] == "bubble"
                has_text_free = has_text_free or text_detection["class_name"] == "text_free"

        affected = []
        decorative_conflicts = []
        for link in region_links:
            if any(
                item["evidence_id"] == evidence_id
                for item in link["kitsumed_links"]
            ):
                affected.append(link["region_id"])
                if link["is_decorative_or_sfx"]:
                    decorative_conflicts.append(link["region_id"])

        reason_codes = ["kitsumed_mask_primary_geometry"]
        if has_bubble:
            reason_codes.append("ogkalu_bubble_support")
        if has_text_bubble:
            reason_codes.append("ogkalu_text_bubble_strengthens_ownership")
        if has_text_free:
            reason_codes.append("ogkalu_text_free_inside_mask_conflict_or_annotation_noise")
        conflict_flags = []
        if decorative_conflicts:
            conflict_flags.append("current_sfx_decorative_region_inside_speech_mask")
        confidence = (
            "high"
            if detection["confidence"] >= 0.70 and (has_bubble or has_text_bubble)
            else "medium"
        )
        downstream = ["text_container_only", "render_constraint_hint"]
        if has_text_bubble:
            downstream.append("ownership_hint")
        if conflict_flags:
            downstream = ["review_only"]
        containers.append(
            {
                "fused_container_id": f"f{index:03d}",
                "page": page,
                "fused_container_type": "speech_bubble",
                "linked_kitsumed_mask_ids": [evidence_id],
                "linked_ogkalu_detection_ids": linked_ogkalu,
                "primary_geometry_source": "kitsumed_mask",
                "bbox": detection.get("bbox_xyxy"),
                "mask_bbox": detection.get("mask_bbox_xyxy"),
                "confidence": confidence,
                "conflict_flags": conflict_flags,
                "reason_codes": reason_codes,
                "affected_current_region_ids": affected,
                "suggested_downstream_use": downstream,
                "would_change_behavior": False,
                "phase4_status": "diagnostic_only",
                "human_review_required": bool(conflict_flags),
            }
        )

    next_index = len(containers)
    for detection in ogkalu:
        evidence_id = detection["model_evidence_id"]
        if evidence_id in ogkalu_claimed:
            continue
        linked_regions = [
            link
            for link in region_links
            if any(
                item["evidence_id"] == evidence_id
                for item in link["ogkalu_links"]
            )
        ]
        region_ids = [link["region_id"] for link in linked_regions]
        has_decorative = any(link["is_decorative_or_sfx"] for link in linked_regions)
        has_caption = any(link["is_caption_or_background"] for link in linked_regions)
        has_speech = any(link["is_speech"] for link in linked_regions)
        class_name = detection["class_name"]
        conflict_flags: list[str] = []
        reason_codes = [f"ogkalu_{class_name}_without_kitsumed_mask"]
        if has_decorative:
            fused_type = "sfx_or_decorative_candidate"
            reason_codes.append("current_pipeline_preserve_role_takes_precedence")
            conflict_flags.append("ogkalu_claims_current_sfx_decorative_region")
            downstream = ["review_only"]
        elif class_name == "text_free":
            fused_type = "caption_or_background_candidate" if has_caption else "free_text"
            downstream = ["missed_text_hint", "review_only"]
            if has_speech:
                conflict_flags.append("text_free_overlaps_current_speech")
        elif class_name == "text_bubble":
            fused_type = "ambiguous"
            downstream = ["ownership_hint", "missed_text_hint", "review_only"]
            if has_speech:
                reason_codes.append("current_speech_region_has_ogkalu_text_bubble_evidence")
        else:
            fused_type = "ambiguous"
            downstream = ["review_only"]
        containers.append(
            {
                "fused_container_id": f"f{next_index:03d}",
                "page": page,
                "fused_container_type": fused_type,
                "linked_kitsumed_mask_ids": [],
                "linked_ogkalu_detection_ids": [evidence_id],
                "primary_geometry_source": "ogkalu_box_advisory",
                "bbox": detection["bbox_xyxy"],
                "mask_bbox": None,
                "confidence": "medium" if detection["confidence"] >= 0.85 else "low",
                "conflict_flags": conflict_flags,
                "reason_codes": reason_codes,
                "affected_current_region_ids": region_ids,
                "suggested_downstream_use": downstream,
                "would_change_behavior": False,
                "phase4_status": "diagnostic_only",
                "human_review_required": True,
            }
        )
        next_index += 1
    return containers


def draw_fusion_overlay(
    image: np.ndarray,
    kitsumed: list[dict[str, Any]],
    ogkalu: list[dict[str, Any]],
    region_links: list[dict[str, Any]],
    containers: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    del region_links, containers
    canvas = image.copy()
    tint = image.copy()
    for detection in kitsumed:
        mask = np.asarray(detection["mask"], dtype=bool)
        tint[mask] = (255, 120, 40)
    canvas = cv2.addWeighted(canvas, 0.72, tint, 0.28, 0.0)
    for detection in kitsumed:
        x0, y0, x1, y1 = [int(round(value)) for value in detection["bbox_xyxy"]]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 100, 30), 3)
    for detection in ogkalu:
        x0, y0, x1, y1 = [int(round(value)) for value in detection["bbox_xyxy"]]
        color = {
            "bubble": (0, 210, 230),
            "text_bubble": (0, 190, 70),
            "text_free": (245, 140, 0),
        }.get(str(detection.get("class_name")), (200, 200, 0))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        canvas,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write fusion overlay: {output_path}")
