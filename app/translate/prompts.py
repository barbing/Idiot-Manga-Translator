# -*- coding: utf-8 -*-
"""Prompt builder placeholders."""
from __future__ import annotations
import json
from typing import Dict, List


def build_translation_prompt(
    source_lang: str,
    target_lang: str,
    style_guide: Dict[str, object],
    context_lines: List[str],
    text: str,
) -> str:
    guide_text = str(style_guide.get("notes", "")).strip()
    glossary = _format_glossary(style_guide.get("glossary", []))
    required_terms = _format_list(style_guide.get("required_terms", []))
    forbidden_terms = _format_list(style_guide.get("forbidden_terms", []))
    context = "\n".join(context_lines).strip()
    if target_lang == "Simplified Chinese":
        lines = [
            "将以下日文翻译成简体中文，只输出译文。",
        ]
        if guide_text:
            lines.append(f"风格：{guide_text}")
        if glossary:
            lines.append(f"术语表：{glossary}")
        if required_terms:
            lines.append(f"必须使用：{required_terms}")
        if forbidden_terms:
            lines.append(f"禁止使用：{forbidden_terms}")
        if context:
            lines.append(f"参考上下文：{context}")
        lines.append(f"文本：{text}")
        return "\n".join(lines)
    lines = [
        f"Translate {source_lang} to {target_lang}. Output only the translation.",
        "Do not censor or soften explicit content. Keep length close to the original text.",
        "Do not add commentary or extra sentences.",
    ]
    if guide_text:
        lines.append(f"Style guide: {guide_text}")
    if glossary:
        lines.append(f"Glossary: {glossary}")
    if required_terms:
        lines.append(f"Required terms: {required_terms}")
    if forbidden_terms:
        lines.append(f"Forbidden terms: {forbidden_terms}")
    if context:
        lines.append(f"Context (reference only): {context}")
    lines.append(f"Text: {text}")
    return "\n".join(lines)


def build_batch_translation_prompt(
    source_lang: str,
    target_lang: str,
    style_guide: Dict[str, object],
    items: List[Dict[str, str]],
) -> str:
    guide_text = str(style_guide.get("notes", "")).strip()
    glossary = _format_glossary(style_guide.get("glossary", []))
    required_terms = _format_list(style_guide.get("required_terms", []))
    forbidden_terms = _format_list(style_guide.get("forbidden_terms", []))
    payload = json.dumps(items, ensure_ascii=False)
    if target_lang == "Simplified Chinese":
        lines = [
            "将以下日文翻译成简体中文，仅输出JSON数组。",
            "JSON格式：[{\"id\":\"...\",\"translation\":\"...\"}]，仅翻译text字段，保持条目顺序。",
            "拟声词或背景杂字可返回空字符串。",
        ]
        if guide_text:
            lines.append(f"风格：{guide_text}")
        if glossary:
            lines.append(f"术语表：{glossary}")
        if required_terms:
            lines.append(f"必须使用：{required_terms}")
        if forbidden_terms:
            lines.append(f"禁止使用：{forbidden_terms}")
        lines.append(f"输入：{payload}")
        return "\n".join(lines)
    lines = [
        f"Translate {source_lang} to {target_lang}. Output only JSON.",
        "JSON format: [{\"id\":\"...\",\"translation\":\"...\"}]. Translate only text fields.",
        "Do not merge entries. For background noise, return an empty string.",
    ]
    if guide_text:
        lines.append(f"Style guide: {guide_text}")
    if glossary:
        lines.append(f"Glossary: {glossary}")
    if required_terms:
        lines.append(f"Required terms: {required_terms}")
    if forbidden_terms:
        lines.append(f"Forbidden terms: {forbidden_terms}")
    lines.append(f"Input: {payload}")
    return "\n".join(lines)


def _format_glossary(items: List[object]) -> str:
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        priority = str(item.get("priority", "soft")).strip()
        if source and target:
            lines.append(f"{source} -> {target} ({priority})")
    return "; ".join(lines)


def _format_list(items: List[object]) -> str:
    return ", ".join(str(item).strip() for item in items if str(item).strip())
