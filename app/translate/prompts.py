# -*- coding: utf-8 -*-
"""Prompt builder placeholders."""
from __future__ import annotations
import json
import re
from typing import Dict, List


_ZH_SOURCE_SYMBOL_CONSERVATION = (
    "原文中作者写出的标点、停顿、强调和表现性符号属于翻译内容。"
    "翻译每个text时，不得遗漏、合并、拆分或擅自增减这些符号；"
    "保持其顺序和重复数量，并使用简体中文排版中自然的等价字形。"
    "尤其要保留三点/六点省略号、!!!、!?、波浪线，以及原文实际使用的破折号（如—或―）。"
)
_ZH_JAPANESE_CHOUONPU_HANDLING = (
    "日文长音符号ー属于日语的长音或口语拖长表达，不是破折号，"
    "也不属于破折号或表现性符号的数量核对对象。"
    "包含ー的完整原文应按语义和中文对白习惯自然翻译；"
    "不得按ー的字符数量、方向或视觉长度强行添加中文破折号或波浪线。"
)
_ZH_TARGET_PUNCTUATION_ALLOWANCE = (
    "可以为中文语法补充普通标点，但不得因此删除原文已有的表现性符号。"
)
_ZH_SOURCE_SYMBOL_SELF_CHECK = (
    "生成每条译文前，请在内部核对原文中的每一个表现性符号连续段；"
    "每个连续段都必须在译文的对应语气位置出现，且自然等价形式、先后顺序和实际重复数量必须完整。"
    "即使压缩文字语义，也不得删除夹在文字之间或位于句首、句末的"
    "省略号、感叹问号、波浪线或原文实际使用的破折号。"
    "例如，“文字……文字”的译文中仍必须在对应两部分之间保留同一省略号连续段。"
)
_GENERAL_SOURCE_SYMBOL_CONSERVATION = (
    "Treat source-authored expressive punctuation and symbols as translation content. "
    "Preserve their order and repetition count, using natural target-language "
    "equivalent glyphs when needed. "
    "Do not mask, replace, merge, split, or silently omit them. "
    "Ordinary punctuation may be added for target-language grammar, but it must "
    "not replace or delete a source-authored expressive symbol."
)
_GENERAL_SOURCE_SYMBOL_SELF_CHECK = (
    " Before returning each translation, silently verify every source symbol run "
    "against the translation, including runs between words and at either edge."
)
_SOURCE_SYMBOL_RUN_RE = re.compile(
    r"\.{3,}|．{3,}|[…‥]+|[・･]{3,}|[!！?？]{2,}|[~～〜]+|[—―‐‑‒–]+"
)


def _source_symbol_conservation_instruction(
    source_lang: str,
    target_lang: str,
) -> str:
    if target_lang == "Simplified Chinese":
        parts = [_ZH_SOURCE_SYMBOL_CONSERVATION]
        if source_lang == "Japanese":
            parts.append(_ZH_JAPANESE_CHOUONPU_HANDLING)
        parts.append(_ZH_TARGET_PUNCTUATION_ALLOWANCE)
        parts.append(_ZH_SOURCE_SYMBOL_SELF_CHECK)
        return "".join(parts)
    return (
        _GENERAL_SOURCE_SYMBOL_CONSERVATION
        + _GENERAL_SOURCE_SYMBOL_SELF_CHECK
    )


def _source_symbol_audit_rows(
    items: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in items:
        item_id = str(item.get("id", "") or "")
        text = str(item.get("text", "") or "")
        required_runs: List[Dict[str, object]] = []
        for match in _SOURCE_SYMBOL_RUN_RE.finditer(text):
            run = match.group(0)
            if all(ch in ".．…‥・･" for ch in run):
                dot_count = sum(
                    3 if ch == "…" else 2 if ch == "‥" else 1
                    for ch in run
                )
                required_runs.append(
                    {
                        "kind": "ellipsis",
                        "dot_count": dot_count,
                        "source_run": run,
                    }
                )
                continue
            if all(ch in "!！?？" for ch in run):
                required_runs.append(
                    {
                        "kind": "emphasis",
                        "sequence": "".join(
                            "!" if ch in "!！" else "?"
                            for ch in run
                        ),
                    }
                )
                continue
            if all(ch in "~～〜" for ch in run):
                required_runs.append(
                    {
                        "kind": "wave",
                        "count": len(run),
                        "source_run": run,
                    }
                )
                continue
            required_runs.append(
                {
                    "kind": "dash",
                    "count": len(run),
                    "source_run": run,
                }
            )
        if required_runs:
            rows.append(
                {
                    "id": item_id,
                    "required_runs_in_order": required_runs,
                }
            )
    return rows


def _source_symbol_checklist_lines(
    target_lang: str,
    items: List[Dict[str, str]],
) -> List[str]:
    rows = _source_symbol_audit_rows(items)
    if not rows:
        return []
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if target_lang == "Simplified Chinese":
        return [
            f"符号核对清单（仅用于核对，不替代或改写输入text）：{payload}",
            "输出前逐项核对：每个translation必须按清单顺序保留标准中文形式和准确点数/符号数量；"
            "三点省略号写作…，六点省略号写作……。清单未列出的字符不得据此新增符号。",
        ]
    return [
        f"Source-symbol checklist (verification only; it does not replace or rewrite text): {payload}",
        "Before output, preserve each listed run in order with the exact dot or symbol count. "
        "Do not add a symbol merely because an item has no listed run.",
    ]


def build_translation_prompt(
    source_lang: str,
    target_lang: str,
    style_guide: Dict[str, object],
    context_lines: List[str],
    text: str,
) -> str:
    guide_text = str(style_guide.get("notes", "")).strip()
    glossary = _format_glossary(style_guide.get("glossary", []))
    characters = _format_characters(style_guide.get("characters", []))
    required_terms = _format_list(style_guide.get("required_terms", []))
    forbidden_terms = _format_list(style_guide.get("forbidden_terms", []))
    context = "\n".join(context_lines).strip()
    symbol_contract = _source_symbol_conservation_instruction(
        source_lang,
        target_lang,
    )
    symbol_checklist = _source_symbol_checklist_lines(
        target_lang,
        [{"id": "single", "text": text}],
    )
    if target_lang == "Simplified Chinese":
        lines = [
            "将以下日文翻译成简体中文，只输出译文。",
            "译文要像中文漫画对白，自然、简洁，不要照搬日语语序。",
            "不要擅自补充主语、解释、旁白或额外句子。",
            "短句就短译；原文是停顿、迟疑、感叹或省略句时，也保持同样的语气，不要补成完整书面句。",
            "不要为了自然度硬加人称、主语或称呼。",
            "人名和称呼优先遵循术语表；敬语请用自然中文处理，不要生硬直译。",
            "长度尽量接近原文，避免过度扩写。",
            symbol_contract,
        ]
        if guide_text:
            lines.append(f"风格：{guide_text}")
        if glossary:
            lines.append(f"术语表：{glossary}")
        if characters:
            lines.append(f"角色设定：{characters}")
        if required_terms:
            lines.append(f"必须使用：{required_terms}")
        if forbidden_terms:
            lines.append(f"禁止使用：{forbidden_terms}")
        if context:
            lines.append(f"参考上下文：{context}")
        lines.extend(symbol_checklist)
        lines.append(f"文本：{text}")
        return "\n".join(lines)
    lines = [
        f"Translate {source_lang} to {target_lang}. Output only the translation.",
        "Do not censor or soften explicit content. Keep length close to the original text.",
        "Do not add commentary or extra sentences.",
        symbol_contract,
    ]
    if guide_text:
        lines.append(f"Style guide: {guide_text}")
    if glossary:
        lines.append(f"Glossary: {glossary}")
    if characters:
        lines.append(f"Characters: {characters}")
    if required_terms:
        lines.append(f"Required terms: {required_terms}")
    if forbidden_terms:
        lines.append(f"Forbidden terms: {forbidden_terms}")
    if context:
        lines.append(f"Context (reference only): {context}")
    lines.extend(symbol_checklist)
    lines.append(f"Text: {text}")
    return "\n".join(lines)


def build_batch_translation_prompt(
    source_lang: str,
    target_lang: str,
    style_guide: Dict[str, object],
    items: List[Dict[str, str]],
    context_lines: List[str] | None = None,
    json_object_wrapper: bool = False,
) -> str:
    guide_text = str(style_guide.get("notes", "")).strip()
    glossary = _format_glossary(style_guide.get("glossary", []))
    characters = _format_characters(style_guide.get("characters", []))
    required_terms = _format_list(style_guide.get("required_terms", []))
    forbidden_terms = _format_list(style_guide.get("forbidden_terms", []))
    context = "\n".join(context_lines or []).strip()
    payload = json.dumps(items, ensure_ascii=False)
    symbol_contract = _source_symbol_conservation_instruction(
        source_lang,
        target_lang,
    )
    symbol_checklist = _source_symbol_checklist_lines(
        target_lang,
        items,
    )
    if target_lang == "Simplified Chinese":
        if json_object_wrapper:
            lines = [
                "将以下日文翻译成简体中文，仅输出有效的 json 对象。",
                "json格式：{\"translations\":[{\"id\":\"...\",\"translation\":\"...\"}]}，仅翻译text字段，保持条目顺序。",
                "不要输出顶层JSON数组，不要输出Markdown，不要解释。",
                "注意：以下文本为同一页漫画的连续对话，请保持语境连贯和人称一致。",
                "拟声词或背景杂字可返回空字符串。",
                "译文要像中文漫画对白，简洁自然，不要照搬日语语序。",
                "不要擅自补充主语、说明或额外句子。",
                "短句就短译；原文是停顿、迟疑、感叹或省略句时，也保持同样的语气，不要补成完整书面句。",
                "不要为了自然度硬加人称、主语或称呼。",
                "人名和称呼优先遵循术语表；敬语请用自然中文处理。",
                symbol_contract,
            ]
            if guide_text:
                lines.append(f"风格：{guide_text}")
            if glossary:
                lines.append(f"术语表：{glossary}")
            if characters:
                lines.append(f"角色设定：{characters}")
            if required_terms:
                lines.append(f"必须使用：{required_terms}")
            if forbidden_terms:
                lines.append(f"禁止使用：{forbidden_terms}")
            if context:
                lines.append(f"参考上下文：{context}")
            lines.extend(symbol_checklist)
            lines.append(f"输入：{payload}")
            return "\n".join(lines)
        lines = [
            "将以下日文翻译成简体中文，仅输出JSON数组。",
            "JSON格式：[{\"id\":\"...\",\"translation\":\"...\"}]，仅翻译text字段，保持条目顺序。",
            "注意：以下文本为同一页漫画的连续对话，请保持语境连贯和人称一致。",
            "拟声词或背景杂字可返回空字符串。",
            "译文要像中文漫画对白，简洁自然，不要照搬日语语序。",
            "不要擅自补充主语、说明或额外句子。",
            "短句就短译；原文是停顿、迟疑、感叹或省略句时，也保持同样的语气，不要补成完整书面句。",
            "不要为了自然度硬加人称、主语或称呼。",
            "人名和称呼优先遵循术语表；敬语请用自然中文处理。",
            symbol_contract,
        ]
        if guide_text:
            lines.append(f"风格：{guide_text}")
        if glossary:
            lines.append(f"术语表：{glossary}")
        if characters:
            lines.append(f"角色设定：{characters}")
        if required_terms:
            lines.append(f"必须使用：{required_terms}")
        if forbidden_terms:
            lines.append(f"禁止使用：{forbidden_terms}")
        if context:
            lines.append(f"参考上下文：{context}")
        lines.extend(symbol_checklist)
        lines.append(f"输入：{payload}")
        return "\n".join(lines)
    if json_object_wrapper:
        lines = [
            f"Translate {source_lang} to {target_lang}. Output only a valid json object.",
            "json format: {\"translations\":[{\"id\":\"...\",\"translation\":\"...\"}]}. Translate only text fields.",
            "Do not output a top-level JSON array. Do not add markdown or explanations.",
            "Do not merge entries. For background noise, return an empty string.",
            symbol_contract,
        ]
        if guide_text:
            lines.append(f"Style guide: {guide_text}")
        if glossary:
            lines.append(f"Glossary: {glossary}")
        if characters:
            lines.append(f"Characters: {characters}")
        if required_terms:
            lines.append(f"Required terms: {required_terms}")
        if forbidden_terms:
            lines.append(f"Forbidden terms: {forbidden_terms}")
        if context:
            lines.append(f"Context (reference only): {context}")
        lines.extend(symbol_checklist)
        lines.append(f"Input: {payload}")
        return "\n".join(lines)
    lines = [
        f"Translate {source_lang} to {target_lang}. Output only JSON.",
        "JSON format: [{\"id\":\"...\",\"translation\":\"...\"}]. Translate only text fields.",
        "Do not merge entries. For background noise, return an empty string.",
        symbol_contract,
    ]
    if guide_text:
        lines.append(f"Style guide: {guide_text}")
    if glossary:
        lines.append(f"Glossary: {glossary}")
    if characters:
        lines.append(f"Characters: {characters}")
    if required_terms:
        lines.append(f"Required terms: {required_terms}")
    if forbidden_terms:
        lines.append(f"Forbidden terms: {forbidden_terms}")
    if context:
        lines.append(f"Context (reference only): {context}")
    lines.extend(symbol_checklist)
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
        
        if not source or not target:
            continue
        
        # Sanitize target: skip entries that contain sentence-style content
        # These would confuse the translation model
        skip_phrases = ["的另一种叫法", "的叫法", "这是", "另一種叫法", "的簡稱"]
        if any(phrase in target for phrase in skip_phrases):
            continue
        
        # Skip targets that are too long (likely explanatory sentences)
        if len(target) > len(source) * 3 and len(target) > 10:
            continue
        
        # Strip trailing punctuation that might have slipped through
        target = target.rstrip("。.，,")
        
        if source and target:
            lines.append(f"{source} -> {target} ({priority})")
    return "; ".join(lines)


def _format_characters(items: List[object]) -> str:
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        gender = str(item.get("gender", "")).strip()
        info = str(item.get("info", "")).strip()
        
        part = name
        if gender:
            part += f" ({gender})"
        if info:
            part += f": {info}"
            
        lines.append(part)
    return "; ".join(lines)



def build_entity_extraction_prompt(
    text_block: str,
    source_lang: str = "Japanese",
    target_lang: str = "Simplified Chinese",
) -> str:
    """Build a prompt to discover and extract entities from text. Adapts to target language."""
    
    if target_lang == "Simplified Chinese":
        # Chinese Instructions for JP->CN models (like Sakura)
        prompt = (
            f"分析以下{source_lang}漫画文本。\n"
            "识别并提取需要统一翻译的重要专有名词（实体）。\n"
            "类别：\n"
            "- Person: 人名（包括昵称、称谓）。\n"
            "- Location: 地名、地标。\n"
            "- Organization: 组织、学校、流派。\n"
            "- Technique: 招式、技能、魔法。\n"
            "- Object: 特殊道具、武器、神器。\n"
            "\n"
            "指令：\n"
            "1. 提取原文中出现的词汇。\n"
            "2. 提供'规范名(Canonical)'（例如：即使文中是昵称，也应映射到全名）。\n"
            "3.将其翻译为简体中文。\n"
            "4. 仅输出一个JSON数组。严禁输出任何思考过程、解释或Markdown标记。\n"
            "\n"
            "示例：\n"
            "[\n"
            "  {\"text\": \"ナルト\", \"type\": \"person\", \"canonical\": \"うずまきナルト\", \"translation\": \"鸣人\", \"info\": \"主角\"},\n"
            "  {\"text\": \"木ノ葉\", \"type\": \"location\", \"canonical\": \"木ノ葉隠れの里\", \"translation\": \"木叶村\", \"info\": \"忍者村\"}\n"
            "]\n"
            "\n"
            f"待分析文本：\n{text_block}\n"
            "\n"
            "JSON格式：\n"
            "[\n"
            "  {\"text\": \"原文\", \"type\": \"person|...\", \"canonical\": \"规范名\", \"translation\": \"中译\", \"info\": \"简要备注\"}\n"
            "]\n"
            "一定要确保以 [ 开头，以 ] 结尾。不要输出 ```json。"
        )
        return prompt

    # Default / Fallback Instructions (English Base)
    prompt = (
        f"Analyze the following {source_lang} text.\n"
        f"Extract entities (Person, Location, Organization) and translate them into {target_lang}.\n"
        f"Text:\n{text_block}\n\n"
        f"Requirements:\n"
        f"1. Output valid JSON list.\n"
        f"2. Fields: \"source\" (original), \"target\" ({target_lang} translation), \"type\".\n"
        f"3. No markdown blocks.\n"
        f"Target JSON Format:\n"
        f"[{{ \"source\": \"...\", \"target\": \"...\", \"type\": \"Person\" }}]"
    )
    return prompt





def _format_list(items: List[str]) -> str:
    valid_items = [str(item).strip() for item in items if item]
    return ", ".join(valid_items)
