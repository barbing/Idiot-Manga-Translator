# -*- coding: utf-8 -*-
"""Project-owned target presentation policy for renderer consumers.

The policy depends only on the configured target language. It never inspects
translated text, source OCR, geometry, fit results, or rendered pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TARGET_PRESENTATION_POLICY_VERSION = "target_presentation_policy_v1"


@dataclass(frozen=True)
class TargetPresentationPolicy:
    policy_id: str
    target_language: str
    target_script: str
    shaping_locale: str
    block_mode_policy: str
    optical_profile_key: str
    measured_fallback_size_policy: str
    automatic_domain_policy: str
    editable_domain_policy: str
    contract_version: str = TARGET_PRESENTATION_POLICY_VERSION

    def to_contract_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "policy_id": self.policy_id,
            "target_language": self.target_language,
            "target_script": self.target_script,
            "shaping_locale": self.shaping_locale,
            "block_mode_policy": self.block_mode_policy,
            "optical_profile_key": self.optical_profile_key,
            "measured_fallback_size_policy": (
                self.measured_fallback_size_policy
            ),
            "automatic_domain_policy": self.automatic_domain_policy,
            "editable_domain_policy": self.editable_domain_policy,
        }


_CJK_POLICY = TargetPresentationPolicy(
    policy_id="target-presentation:zh-Hans:v1",
    target_language="zh-Hans",
    target_script="Hani",
    shaping_locale="zh-Hans",
    block_mode_policy="preserve_source",
    optical_profile_key="cjk",
    measured_fallback_size_policy="upper_supported_non_decreasing",
    automatic_domain_policy="source_parent",
    editable_domain_policy="authorized_speech_container_or_source",
)

_ENGLISH_POLICY = TargetPresentationPolicy(
    policy_id="target-presentation:en:v1",
    target_language="en",
    target_script="Latn",
    shaping_locale="en",
    block_mode_policy="horizontal",
    optical_profile_key="latin",
    measured_fallback_size_policy="upper_supported_non_decreasing",
    automatic_domain_policy=(
        "source_side_anchored_speech_container_or_source"
    ),
    editable_domain_policy="authorized_speech_container_or_source",
)

_ALIASES = {
    "simplified chinese": _CJK_POLICY,
    "zh": _CJK_POLICY,
    "zh-cn": _CJK_POLICY,
    "zh-hans": _CJK_POLICY,
    "english": _ENGLISH_POLICY,
    "en": _ENGLISH_POLICY,
    "en-us": _ENGLISH_POLICY,
}


def target_presentation_policy(target_language: str) -> TargetPresentationPolicy:
    normalized = str(target_language or "").strip().lower().replace("_", "-")
    policy = _ALIASES.get(normalized)
    if policy is None:
        raise ValueError(f"unsupported target language: {target_language}")
    return policy
