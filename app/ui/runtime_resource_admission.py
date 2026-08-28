"""Typed, model-free memory admission for one immutable GUI run candidate.

The estimator never loads a model and never rewrites run settings.  It measures
current host/GPU availability, estimates the selected local runtime residency,
and returns a fail-closed receipt for the GUI Start gate.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import re
import struct
import subprocess
from typing import Callable, Mapping
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from app.config.settings_contracts import canonical_fingerprint
from app.models.resolution import (
    models_root,
    resolve_cleanup_inpaint_model_file,
    resolve_kitsumed_speech_bubble_model,
    resolve_llama_server_executable,
    resolve_manga_ocr_local_dir,
    resolve_manga_ocr_system_ref,
    resolve_ogkalu_text_bubble_model,
    resolve_paddle_ocr_vl_mmproj_file,
    resolve_paddle_ocr_vl_model_file,
    resolve_yuzumarker_font_onnx_file,
)
from app.platform_services.compute import (
    ComputeCapabilitySnapshot,
    probe_llama_cpp_python_backend,
    probe_llama_server_backend,
    probe_mps_memory,
    select_onnx_providers,
    select_torch_device,
)
from app.platform_services.contracts import ComputeBackend


GIB = 1024**3
MIB = 1024**2


class ResourceAdmissionStatus(str, Enum):
    SAFE = "safe"
    RISK = "risk"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"


class ResourceMonitorLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RuntimeGpuMemoryDevice:
    index: str
    uuid: str
    name: str
    total_bytes: int
    available_bytes: int

    def __post_init__(self) -> None:
        if self.total_bytes <= 0:
            raise ValueError("GPU total memory must be positive")
        if not 0 <= self.available_bytes <= self.total_bytes:
            raise ValueError("GPU available memory is outside total memory")


@dataclass(frozen=True)
class RuntimeMemorySnapshot:
    total_ram_bytes: int
    available_ram_bytes: int
    gpu_name: str = ""
    total_vram_bytes: int | None = None
    available_vram_bytes: int | None = None
    gpu_devices: tuple[RuntimeGpuMemoryDevice, ...] = ()
    source: str = ""
    detail: str = ""
    backend: ComputeBackend = ComputeBackend.CPU
    onnx_backend: ComputeBackend = ComputeBackend.CPU
    llama_server_backend: ComputeBackend = ComputeBackend.CPU
    llama_cpp_backend: ComputeBackend = ComputeBackend.CPU
    unified_memory: bool = False

    def __post_init__(self) -> None:
        if self.total_ram_bytes <= 0:
            raise ValueError("total_ram_bytes must be positive")
        if not 0 <= self.available_ram_bytes <= self.total_ram_bytes:
            raise ValueError("available_ram_bytes is outside total RAM")
        if (self.total_vram_bytes is None) != (self.available_vram_bytes is None):
            raise ValueError("GPU total and available memory must be supplied together")
        if self.total_vram_bytes is not None:
            if self.total_vram_bytes <= 0:
                raise ValueError("total_vram_bytes must be positive")
            if not 0 <= int(self.available_vram_bytes or 0) <= self.total_vram_bytes:
                raise ValueError("available_vram_bytes is outside total VRAM")
        if any(not isinstance(item, RuntimeGpuMemoryDevice) for item in self.gpu_devices):
            raise TypeError("gpu_devices must contain RuntimeGpuMemoryDevice values")
        if not isinstance(self.backend, ComputeBackend):
            raise TypeError("backend must be ComputeBackend")
        if not isinstance(self.onnx_backend, ComputeBackend):
            raise TypeError("onnx_backend must be ComputeBackend")
        if not isinstance(self.llama_server_backend, ComputeBackend):
            raise TypeError("llama_server_backend must be ComputeBackend")
        if not isinstance(self.llama_cpp_backend, ComputeBackend):
            raise TypeError("llama_cpp_backend must be ComputeBackend")
        if self.unified_memory and self.total_vram_bytes is None:
            raise ValueError("unified-memory snapshots require a working-set limit")
        if self.unified_memory and self.backend is ComputeBackend.CUDA:
            raise ValueError("CUDA snapshots use a separate VRAM budget")


@dataclass(frozen=True)
class RuntimeResourceAssets:
    translation_model_label: str = ""
    translation_model_bytes: int = 0
    translation_block_count: int | None = None
    translation_shard_count: int = 1
    translation_expert_count: int = 0
    translation_kv_bytes_per_token: int = 0
    translation_model_already_resident: bool = False
    translation_resident_model_bytes: int = 0
    translation_resident_vram_bytes: int = 0
    translation_remote: bool = False
    translation_context_bytes: int = 0
    translation_vision_workspace_bytes: int = 0
    translation_incremental_gpu_bytes: int | None = None
    translation_incremental_ram_bytes: int | None = None
    translation_residency_method: str = ""
    paddle_model_bytes: int = 0
    paddle_mmproj_bytes: int = 0
    manga_ocr_model_bytes: int = 0
    ner_model_bytes: int = 0
    detector_model_bytes: int = 0
    bubble_model_bytes: int = 0
    cleanup_model_bytes: int = 0
    font_model_bytes: int = 0
    unresolved: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "translation_model_bytes",
            "paddle_model_bytes",
            "paddle_mmproj_bytes",
            "manga_ocr_model_bytes",
            "ner_model_bytes",
            "detector_model_bytes",
            "bubble_model_bytes",
            "cleanup_model_bytes",
            "font_model_bytes",
            "translation_context_bytes",
            "translation_vision_workspace_bytes",
            "translation_resident_model_bytes",
            "translation_resident_vram_bytes",
            "translation_kv_bytes_per_token",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must not be negative")
        if self.translation_block_count is not None and self.translation_block_count <= 0:
            raise ValueError("translation_block_count must be positive")
        if self.translation_shard_count <= 0:
            raise ValueError("translation_shard_count must be positive")
        if self.translation_expert_count < 0:
            raise ValueError("translation_expert_count must not be negative")
        for field_name in (
            "translation_incremental_gpu_bytes",
            "translation_incremental_ram_bytes",
        ):
            value = getattr(self, field_name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{field_name} must not be negative")


@dataclass(frozen=True)
class OllamaRuntimeInventory:
    endpoint: str
    model: str
    model_bytes: int = 0
    resident: bool = False
    resident_model_bytes: int = 0
    resident_vram_bytes: int = 0
    configured_context_tokens: int = 0
    resident_context_tokens: int = 0
    context_bytes: int = 0
    vision_workspace_bytes: int = 0
    remote: bool = False
    unresolved: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GgufMetadata:
    architecture: str = ""
    block_count: int | None = None
    expert_count: int = 0
    split_count: int = 1
    split_index: int = 0
    tensor_count: int = 0
    embedding_length: int = 0
    kv_bytes_per_token: int = 0

    @property
    def automatic_layout_supported(self) -> bool:
        return bool(
            self.architecture
            and self.block_count is not None
            and self.block_count > 0
            and self.expert_count == 0
        )


@dataclass(frozen=True)
class ResourceComponentEstimate:
    component_id: str
    label: str
    gpu_bytes: int
    ram_bytes: int
    method: str
    backend: ComputeBackend = ComputeBackend.CPU

    def __post_init__(self) -> None:
        if not self.component_id.strip() or not self.label.strip():
            raise ValueError("resource component identity is required")
        if self.gpu_bytes < 0 or self.ram_bytes < 0:
            raise ValueError("resource component estimates must not be negative")
        if not isinstance(self.backend, ComputeBackend):
            raise TypeError("resource component backend must be ComputeBackend")
        if self.backend is ComputeBackend.CPU and self.gpu_bytes:
            raise ValueError("CPU components cannot reserve accelerator memory")


@dataclass(frozen=True)
class RuntimeResourceAdmissionReport:
    status: ResourceAdmissionStatus
    settings_fingerprint: str
    pipeline_values_fingerprint: str
    effective_pipeline_values_fingerprint: str
    checked_at_utc: str
    gpu_name: str
    total_vram_bytes: int | None
    available_vram_bytes: int | None
    projected_vram_bytes: int
    vram_reserve_bytes: int
    vram_budget_bytes: int
    total_ram_bytes: int
    available_ram_bytes: int
    projected_ram_bytes: int
    ram_reserve_bytes: int
    ram_budget_bytes: int
    components: tuple[ResourceComponentEstimate, ...]
    reasons: tuple[str, ...]
    actions: tuple[str, ...]
    runtime_overrides: tuple[tuple[str, object], ...] = ()
    warnings: tuple[str, ...] = ()
    recommended_changes: tuple[tuple[str, object], ...] = ()
    backend: ComputeBackend = ComputeBackend.CPU
    onnx_backend: ComputeBackend = ComputeBackend.CPU
    llama_server_backend: ComputeBackend = ComputeBackend.CPU
    llama_cpp_backend: ComputeBackend = ComputeBackend.CPU
    unified_memory: bool = False
    projected_unified_bytes: int = 0
    unified_reserve_bytes: int = 0
    unified_budget_bytes: int = 0

    @property
    def accelerator_backends(self) -> tuple[ComputeBackend, ...]:
        order = {
            ComputeBackend.CUDA: 0,
            ComputeBackend.MPS: 1,
            ComputeBackend.COREML: 2,
            ComputeBackend.METAL: 3,
            ComputeBackend.CPU: 4,
        }
        selected = {
            item.backend
            for item in self.components
            if item.gpu_bytes and item.backend is not ComputeBackend.CPU
        }
        return tuple(sorted(selected, key=order.__getitem__))

    @property
    def accelerator_label(self) -> str:
        backends = self.accelerator_backends
        if not backends:
            return "CPU"
        names = "/".join(
            {
                ComputeBackend.CUDA: "CUDA",
                ComputeBackend.MPS: "MPS",
                ComputeBackend.COREML: "CoreML",
                ComputeBackend.METAL: "Metal",
                ComputeBackend.CPU: "CPU",
            }[item]
            for item in backends
        )
        return f"{names} unified memory" if self.unified_memory else f"{names} VRAM"

    @property
    def safe_to_start(self) -> bool:
        return self.status is ResourceAdmissionStatus.SAFE

    @property
    def label(self) -> str:
        return {
            ResourceAdmissionStatus.SAFE: "Memory budget safe",
            ResourceAdmissionStatus.RISK: "Memory budget risk",
            ResourceAdmissionStatus.BLOCKED: "Memory budget exceeded",
            ResourceAdmissionStatus.UNAVAILABLE: "Memory budget unavailable",
        }[self.status]

    @property
    def tone(self) -> str:
        return {
            ResourceAdmissionStatus.SAFE: "ready",
            ResourceAdmissionStatus.RISK: "warning",
            ResourceAdmissionStatus.BLOCKED: "error",
            ResourceAdmissionStatus.UNAVAILABLE: "error",
        }[self.status]

    @property
    def detail(self) -> str:
        auto_layers = next(
            (
                int(value)
                for key, value in self.runtime_overrides
                if key == "gguf_n_gpu_layers"
            ),
            None,
        )
        auto_prefix = (
            (
                "GGUF Automatic selected CPU translation for this run. "
                if auto_layers == 0
                else f"GGUF Automatic resolved to {auto_layers} GPU layers for this run. "
            )
            if auto_layers is not None
            else ""
        )
        warning_suffix = f" {self.warnings[0]}" if self.warnings else ""
        if self.status is ResourceAdmissionStatus.UNAVAILABLE:
            reason = (
                self.reasons[0]
                if self.reasons
                else "Accelerator memory could not be measured."
            )
            return (
                f"{auto_prefix}{reason} Start is blocked until the local memory "
                "budget can be verified."
            )
        ram_pressure = any(
            "host residency" in reason
            or "system-memory" in reason
            or "unified-memory" in reason
            for reason in self.reasons
        )
        if ram_pressure and self.status in {
            ResourceAdmissionStatus.BLOCKED,
            ResourceAdmissionStatus.RISK,
        }:
            projected = _format_bytes(
                self.projected_unified_bytes
                if self.unified_memory
                else self.projected_ram_bytes
            )
            available = _format_bytes(self.available_ram_bytes)
            reserve = _format_bytes(
                self.unified_reserve_bytes
                if self.unified_memory
                else self.ram_reserve_bytes
            )
            memory_label = (
                f"{self.accelerator_label} and system memory"
                if self.unified_memory
                else "system memory"
            )
            if self.status is ResourceAdmissionStatus.BLOCKED:
                return auto_prefix + (
                    f"{memory_label}: needs {projected}; {available} is available. "
                    "Start is blocked."
                )
            return auto_prefix + (
                f"{memory_label}: needs {projected} plus {reserve} reserve; "
                f"{available} is available. Start is blocked."
            )
        if self.available_vram_bytes is not None and self.projected_vram_bytes:
            projected = _format_bytes(self.projected_vram_bytes)
            available = _format_bytes(self.available_vram_bytes)
            reserve = _format_bytes(self.vram_reserve_bytes)
            if self.status is ResourceAdmissionStatus.BLOCKED:
                return auto_prefix + (
                    f"{self.accelerator_label}: needs {projected}; "
                    f"{available} is available. "
                    "Start is blocked."
                )
            if self.status is ResourceAdmissionStatus.RISK:
                return auto_prefix + (
                    f"{self.accelerator_label}: needs {projected} plus {reserve} "
                    f"reserve; {available} is available. Start is blocked."
                )
            return auto_prefix + (
                f"{self.accelerator_label}: {projected} estimated · "
                f"{available} available · {reserve} reserved."
                f"{warning_suffix}"
            )
        if self.status is ResourceAdmissionStatus.BLOCKED:
            return auto_prefix + (
                f"Estimated host-memory demand ({_format_bytes(self.projected_ram_bytes)}) "
                f"exceeds currently available RAM ({_format_bytes(self.available_ram_bytes)}). "
                "Start is blocked."
            )
        if self.status is ResourceAdmissionStatus.RISK:
            return auto_prefix + (
                "The selected run would consume the required system-memory safety "
                "reserve. Start is blocked."
            )
        return (
            auto_prefix
            + "The selected run fits the measured system-memory budget."
            + warning_suffix
        )

    @property
    def facts(self) -> tuple[tuple[str, str], ...]:
        facts: list[tuple[str, str]] = []
        if self.available_vram_bytes is not None and self.projected_vram_bytes:
            auto_layers = next(
                (
                    int(value)
                    for key, value in self.runtime_overrides
                    if key == "gguf_n_gpu_layers"
                ),
                None,
            )
            auto_detail = (
                (
                    " · Automatic CPU"
                    if auto_layers == 0
                    else f" · Automatic {auto_layers} layers"
                )
                if auto_layers is not None
                else ""
            )
            facts.append(
                (
                    f"{self.accelerator_label} "
                    f"{_format_bytes(self.projected_vram_bytes)} estimated · "
                    f"{_format_bytes(self.vram_budget_bytes)} budget{auto_detail}",
                    self.tone,
                )
            )
        facts.append(
            (
                (
                    f"Unified RAM {_format_bytes(self.projected_unified_bytes)} "
                    f"estimated · {_format_bytes(self.unified_budget_bytes)} budget"
                    if self.unified_memory
                    else f"RAM {_format_bytes(self.projected_ram_bytes)} estimated · "
                    f"{_format_bytes(self.ram_budget_bytes)} budget"
                ),
                (
                    "ready"
                    if (
                        self.projected_unified_bytes <= self.unified_budget_bytes
                        if self.unified_memory
                        else self.projected_ram_bytes <= self.ram_budget_bytes
                    )
                    else "error"
                ),
            )
        )
        return tuple(facts[:2])

    def is_fresh(self, *, max_age_seconds: float = 5.0) -> bool:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        try:
            checked = datetime.fromisoformat(
                self.checked_at_utc.replace("Z", "+00:00")
            )
        except ValueError:
            return False
        return (
            datetime.now(timezone.utc) - checked
        ).total_seconds() <= max_age_seconds


@dataclass(frozen=True)
class RuntimeResourceMonitorReport:
    level: ResourceMonitorLevel
    admission_settings_fingerprint: str
    checked_at_utc: str
    available_ram_bytes: int
    available_vram_bytes: int | None
    ram_reserve_bytes: int
    vram_reserve_bytes: int
    reasons: tuple[str, ...] = ()

    @property
    def tone(self) -> str:
        return {
            ResourceMonitorLevel.SAFE: "ready",
            ResourceMonitorLevel.WARNING: "warning",
            ResourceMonitorLevel.CRITICAL: "error",
            ResourceMonitorLevel.UNAVAILABLE: "warning",
        }[self.level]

    @property
    def detail(self) -> str:
        if self.level is ResourceMonitorLevel.SAFE:
            return "Active run remains inside its admitted memory reserve."
        if self.reasons:
            return self.reasons[0]
        return "Active-run memory reserve could not be verified."


def monitor_runtime_reserve(
    *,
    admission: RuntimeResourceAdmissionReport,
    memory: RuntimeMemorySnapshot,
) -> RuntimeResourceMonitorReport:
    """Classify current free memory without changing or stopping the run."""

    if not isinstance(admission, RuntimeResourceAdmissionReport):
        raise TypeError("admission must be RuntimeResourceAdmissionReport")
    if not isinstance(memory, RuntimeMemorySnapshot):
        raise TypeError("memory must be RuntimeMemorySnapshot")
    reasons: list[str] = []
    level = ResourceMonitorLevel.SAFE
    if admission.projected_vram_bytes:
        if memory.available_vram_bytes is None:
            level = ResourceMonitorLevel.UNAVAILABLE
            reasons.append(
                f"Active-run {admission.accelerator_label} reserve cannot be "
                "measured; execution continues unchanged."
            )
        else:
            critical_vram = max(512 * MIB, admission.vram_reserve_bytes // 3)
            if memory.available_vram_bytes < critical_vram:
                level = ResourceMonitorLevel.CRITICAL
                reasons.append(
                    f"Critical {admission.accelerator_label} pressure: less than "
                    f"{_format_bytes(critical_vram)} remains free. Close other "
                    "accelerator-heavy applications; YomiFrame will not alter or "
                    "cancel this run."
                )
            elif memory.available_vram_bytes < admission.vram_reserve_bytes:
                level = ResourceMonitorLevel.WARNING
                reasons.append(
                    f"{admission.accelerator_label} safety reserve has been crossed. "
                    "Close other accelerator-heavy applications; YomiFrame will not "
                    "alter this run."
                )
    critical_ram = max(GIB, admission.ram_reserve_bytes // 3)
    if memory.available_ram_bytes < critical_ram:
        level = ResourceMonitorLevel.CRITICAL
        reasons.insert(
            0,
            "Critical system-memory pressure: less than "
            f"{_format_bytes(critical_ram)} remains available. Close other "
            "applications; YomiFrame will not alter or cancel this run.",
        )
    elif (
        memory.available_ram_bytes < admission.ram_reserve_bytes
        and level is ResourceMonitorLevel.SAFE
    ):
        level = ResourceMonitorLevel.WARNING
        reasons.append(
            "System-memory safety reserve has been crossed. Close other "
            "applications; YomiFrame will not alter this run."
        )
    return RuntimeResourceMonitorReport(
        level=level,
        admission_settings_fingerprint=admission.settings_fingerprint,
        checked_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        available_ram_bytes=memory.available_ram_bytes,
        available_vram_bytes=memory.available_vram_bytes,
        ram_reserve_bytes=admission.ram_reserve_bytes,
        vram_reserve_bytes=admission.vram_reserve_bytes,
        reasons=tuple(dict.fromkeys(reasons)),
    )


class RuntimeResourceMonitorService:
    def __init__(
        self,
        *,
        memory_probe: Callable[[], RuntimeMemorySnapshot] | None = None,
        compute: ComputeCapabilitySnapshot | None = None,
    ) -> None:
        self._memory_probe = memory_probe or (
            lambda: probe_runtime_memory(compute=compute)
        )

    def sample(
        self,
        admission: RuntimeResourceAdmissionReport,
    ) -> RuntimeResourceMonitorReport:
        try:
            memory = self._memory_probe()
        except Exception:
            return RuntimeResourceMonitorReport(
                level=ResourceMonitorLevel.UNAVAILABLE,
                admission_settings_fingerprint=admission.settings_fingerprint,
                checked_at_utc=datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                available_ram_bytes=0,
                available_vram_bytes=None,
                ram_reserve_bytes=admission.ram_reserve_bytes,
                vram_reserve_bytes=admission.vram_reserve_bytes,
                reasons=(
                    "Active-run memory reserve could not be measured; execution "
                    "continues unchanged.",
                ),
            )
        return monitor_runtime_reserve(admission=admission, memory=memory)


def assess_runtime_resources(
    *,
    settings_fingerprint: str,
    pipeline_values: Mapping[str, object],
    memory: RuntimeMemorySnapshot,
    assets: RuntimeResourceAssets,
) -> RuntimeResourceAdmissionReport:
    """Return one immutable admission decision without mutating settings."""

    normalized_fingerprint = str(settings_fingerprint or "").strip()
    if not normalized_fingerprint:
        raise ValueError("settings_fingerprint is required")
    if not isinstance(memory, RuntimeMemorySnapshot):
        raise TypeError("memory must be RuntimeMemorySnapshot")
    if not isinstance(assets, RuntimeResourceAssets):
        raise TypeError("assets must be RuntimeResourceAssets")

    values = dict(pipeline_values)
    use_gpu = bool(values.get("use_gpu", True))
    torch_backend = _torch_component_backend(memory, use_gpu=use_gpu)
    onnx_backend = _onnx_component_backend(memory, use_gpu=use_gpu)
    detector_backend = _detector_component_backend(memory, use_gpu=use_gpu)
    paddle_backend = _paddle_component_backend(memory, use_gpu=use_gpu)
    components: list[ResourceComponentEstimate] = []
    fixed_components: list[ResourceComponentEstimate] = []

    ocr_gpu = 0
    ocr_ram = int(assets.paddle_model_bytes + assets.paddle_mmproj_bytes)
    if str(values.get("ocr_engine") or "") == "PaddleOCR-VL" and ocr_ram:
        if paddle_backend is not ComputeBackend.CPU and _paddle_gpu_layers() != 0:
            ocr_gpu = ocr_ram
        fixed_components.append(
            ResourceComponentEstimate(
                component_id="ocr_model",
                label="PaddleOCR-VL model and projector",
                gpu_bytes=ocr_gpu,
                ram_bytes=ocr_ram,
                method="measured_local_asset_size",
                backend=(
                    paddle_backend if ocr_gpu else ComputeBackend.CPU
                ),
            )
        )
        ocr_context = _positive_env_int("MT_PADDLEOCR_VL_CTX_SIZE", 4096)
        ocr_slots = _positive_env_int("MT_PADDLEOCR_VL_PARALLEL", 4)
        ocr_kv_bytes = ocr_context * ocr_slots * 64 * 1024
        fixed_components.append(
            ResourceComponentEstimate(
                component_id="ocr_kv_cache",
                label="PaddleOCR-VL context cache",
                gpu_bytes=ocr_kv_bytes if ocr_gpu else 0,
                ram_bytes=ocr_kv_bytes,
                method="configured_context_and_slot_count",
                backend=(
                    paddle_backend if ocr_gpu else ComputeBackend.CPU
                ),
            )
        )
    elif (
        str(values.get("ocr_engine") or "") == "MangaOCR"
        and assets.manga_ocr_model_bytes
    ):
        _append_fixed_component(
            fixed_components,
            component_id="ocr_model",
            label="MangaOCR runtime",
            file_bytes=assets.manga_ocr_model_bytes,
            backend=torch_backend,
            gpu_multiplier=2.0,
        )

    _append_fixed_component(
        fixed_components,
        component_id="detector",
        label="ComicTextDetector runtime",
        file_bytes=assets.detector_model_bytes,
        backend=detector_backend,
        gpu_multiplier=2.0,
    )
    _append_fixed_component(
        fixed_components,
        component_id="bubble_detection",
        label="Bubble-detection runtimes",
        file_bytes=assets.bubble_model_bytes,
        backend=onnx_backend,
        gpu_multiplier=2.0,
    )
    if str(values.get("inpaint_mode") or "") == "ai":
        _append_fixed_component(
            fixed_components,
            component_id="cleanup",
            label="Cleanup runtime",
            file_bytes=assets.cleanup_model_bytes,
            backend=torch_backend,
            gpu_multiplier=2.0,
        )
    if str(values.get("font_detection") or "").casefold() not in {"", "off", "none"}:
        _append_fixed_component(
            fixed_components,
            component_id="font_detection",
            label="Font-detection runtime",
            file_bytes=assets.font_model_bytes,
            backend=onnx_backend,
            gpu_multiplier=1.5,
        )
    if bool(values.get("prescan_use_ner", False)) and assets.ner_model_bytes:
        _append_fixed_component(
            fixed_components,
            component_id="prescan_ner",
            label="Japanese NER runtime",
            file_bytes=assets.ner_model_bytes,
            backend=torch_backend,
            gpu_multiplier=2.0,
        )

    translator_backend = str(values.get("translator_backend") or "").strip()
    local_translation_backend = _translation_component_backend(
        memory,
        use_gpu=use_gpu,
        translator_backend=translator_backend,
        assets=assets,
    )
    gguf_batch_workspace = (
        max(128 * MIB, max(1, int(values.get("gguf_n_batch", 256) or 256)) * MIB)
        if translator_backend == "GGUF"
        else 0
    )
    runtime_overrides: tuple[tuple[str, object], ...] = ()
    effective_values = dict(values)
    resolution_reasons: list[str] = []
    if (
        len(memory.gpu_devices) > 1
        and local_translation_backend is ComputeBackend.CUDA
        and translator_backend in {"GGUF", "Ollama"}
        and not assets.translation_remote
        and (
            translator_backend == "Ollama"
            or int(values.get("gguf_n_gpu_layers", -1) or 0) != 0
        )
    ):
        resolution_reasons.append(
            "Multiple visible GPUs cannot be safely attributed to this local "
            "model plan. Select one device with CUDA_VISIBLE_DEVICES and check again."
        )
    translation_gpu = 0
    translation_ram = int(assets.translation_model_bytes)
    if assets.translation_model_bytes:
        if translator_backend == "GGUF":
            configured_layers = int(values.get("gguf_n_gpu_layers", -1) or 0)
            effective_layers = configured_layers
            if (
                local_translation_backend is ComputeBackend.CPU
                and configured_layers < 0
            ):
                effective_layers = 0
                runtime_overrides = (("gguf_n_gpu_layers", 0),)
                effective_values["gguf_n_gpu_layers"] = 0
            elif (
                local_translation_backend is ComputeBackend.CPU
                and configured_layers > 0
            ):
                resolution_reasons.append(
                    "The installed llama-cpp-python runtime cannot honor explicit "
                    "GPU layers. Set GGUF GPU layers to 0 or install a GPU-capable "
                    "llama-cpp-python build."
                )
            elif assets.translation_expert_count > 0 and configured_layers != 0:
                resolution_reasons.append(
                    "GPU residency for this MoE GGUF layout cannot be safely "
                    "estimated. Select CPU layers (0) or another supported model."
                )
            elif configured_layers < 0:
                if assets.translation_block_count is None:
                    resolution_reasons.append(
                        "The GGUF block count could not be measured for Automatic "
                        "GPU fitting."
                    )
                else:
                    context_tokens = max(
                        512, int(values.get("gguf_n_ctx", 4096) or 4096)
                    )
                    effective_layers = _auto_fit_gguf_layers(
                        model_bytes=assets.translation_model_bytes,
                        block_count=assets.translation_block_count,
                        context_tokens=context_tokens,
                        kv_bytes_per_token=assets.translation_kv_bytes_per_token,
                        fixed_gpu_bytes=(
                            sum(item.gpu_bytes for item in fixed_components)
                            + gguf_batch_workspace
                        ),
                        memory=memory,
                    )
                    runtime_overrides = (("gguf_n_gpu_layers", effective_layers),)
                    effective_values["gguf_n_gpu_layers"] = effective_layers
            fraction = _offload_fraction(
                effective_layers,
                assets.translation_block_count,
            )
            if local_translation_backend is ComputeBackend.CPU:
                fraction = 0.0
            translation_gpu = int(round(assets.translation_model_bytes * fraction))
            # llama-cpp-python keeps ``use_mmap=True`` by default.  GPU-offloaded
            # weight pages do not require an equal committed host copy; budget
            # the non-offloaded fraction and leave mappings/metadata/scratch to
            # the general host allowance below.
            translation_ram = int(
                round(assets.translation_model_bytes * (1.0 - fraction))
            )
        elif translator_backend == "Ollama":
            if assets.translation_remote:
                translation_gpu = 0
                translation_ram = 0
            elif assets.translation_incremental_gpu_bytes is not None:
                translation_gpu = int(assets.translation_incremental_gpu_bytes)
                translation_ram = int(
                    assets.translation_incremental_ram_bytes or 0
                )
            elif assets.translation_model_already_resident:
                translation_gpu = 0
                translation_ram = 0
            elif (
                local_translation_backend is ComputeBackend.CPU
                or memory.available_vram_bytes is None
            ):
                translation_gpu = 0
                translation_ram = int(assets.translation_model_bytes)
            else:
                reserve = max(
                    1536 * MIB,
                    int((memory.total_vram_bytes or 0) * 0.10),
                )
                fixed_gpu = sum(item.gpu_bytes for item in fixed_components)
                context_gpu = (
                    assets.translation_context_bytes
                    + assets.translation_vision_workspace_bytes
                )
                available_for_weights = max(
                    0,
                    int(memory.available_vram_bytes or 0)
                    - reserve
                    - fixed_gpu
                    - context_gpu,
                )
                translation_gpu = min(
                    int(assets.translation_model_bytes),
                    available_for_weights,
                )
                translation_ram = max(
                    0,
                    int(assets.translation_model_bytes) - translation_gpu,
                )
        components.append(
            ResourceComponentEstimate(
                component_id="translation_model",
                label=assets.translation_model_label or "Translation model",
                gpu_bytes=translation_gpu,
                ram_bytes=translation_ram,
                method=(
                    (
                        "measured_file_size_and_auto_fitted_layer_fraction"
                        if runtime_overrides
                        else "measured_file_size_and_configured_layer_fraction"
                    )
                    if translator_backend == "GGUF"
                    else assets.translation_residency_method
                    or "local_runtime_inventory"
                ),
                backend=(
                    local_translation_backend
                    if translation_gpu
                    else ComputeBackend.CPU
                ),
            )
        )
        if translator_backend == "GGUF":
            context_tokens = max(512, int(values.get("gguf_n_ctx", 4096) or 4096))
            block_count = max(1, int(assets.translation_block_count or 32))
            kv_bytes = context_tokens * max(
                64 * 1024,
                int(assets.translation_kv_bytes_per_token or block_count * 4096),
            )
            components.append(
                ResourceComponentEstimate(
                    component_id="translation_kv_cache",
                    label="Translation context cache",
                    gpu_bytes=kv_bytes if translation_gpu else 0,
                    ram_bytes=kv_bytes,
                    method="context_tokens_and_model_block_count",
                    backend=(
                        local_translation_backend
                        if translation_gpu
                        else ComputeBackend.CPU
                    ),
                )
            )
            components.append(
                ResourceComponentEstimate(
                    component_id="translation_batch_workspace",
                    label="GGUF prompt-batch workspace",
                    gpu_bytes=(
                        gguf_batch_workspace
                        if translation_gpu
                        else 0
                    ),
                    ram_bytes=(
                        gguf_batch_workspace
                        if not translation_gpu
                        else 0
                    ),
                    method="configured_prompt_batch_safety_envelope",
                    backend=(
                        local_translation_backend
                        if translation_gpu
                        else ComputeBackend.CPU
                    ),
                )
            )
        elif translator_backend == "Ollama" and (
            assets.translation_context_bytes
            or assets.translation_vision_workspace_bytes
        ):
            context_bytes = (
                assets.translation_context_bytes
                + assets.translation_vision_workspace_bytes
            )
            ollama_context_on_gpu = bool(
                not assets.translation_remote
                and local_translation_backend is not ComputeBackend.CPU
                and memory.available_vram_bytes is not None
                and (
                    not assets.translation_model_already_resident
                    or assets.translation_resident_vram_bytes > 0
                )
            )
            components.append(
                ResourceComponentEstimate(
                    component_id="translation_context",
                    label="Ollama context and vision workspace",
                    gpu_bytes=(
                        context_bytes
                        if ollama_context_on_gpu
                        else 0
                    ),
                    ram_bytes=(
                        context_bytes
                        if not ollama_context_on_gpu
                        and not assets.translation_remote
                        else 0
                    ),
                    method="ollama_model_metadata_and_configured_context",
                    backend=(
                        local_translation_backend
                        if ollama_context_on_gpu
                        else ComputeBackend.CPU
                    ),
                )
            )
    components.extend(fixed_components)

    # ``projected_vram`` is retained for receipt compatibility.  On unified-
    # memory hosts it means the selected accelerator working set, not discrete
    # VRAM; ``projected_unified`` adds the concurrent host residency.
    projected_vram = sum(item.gpu_bytes for item in components)
    # Host mappings, image buffers, Python/Qt state, and non-offloaded weights all
    # consume RAM even when the primary tensors are GPU-resident.
    projected_ram = sum(item.ram_bytes for item in components) + GIB
    projected_unified = (
        GIB + sum(_unified_component_bytes(item) for item in components)
        if memory.unified_memory
        else 0
    )
    vram_reserve = (
        max(1536 * MIB, int((memory.total_vram_bytes or 0) * 0.10))
        if projected_vram
        else 0
    )
    ram_reserve = max(4 * GIB, int(memory.total_ram_bytes * 0.10))
    vram_budget = max(0, int(memory.available_vram_bytes or 0) - vram_reserve)
    ram_budget = max(0, memory.available_ram_bytes - ram_reserve)
    unified_reserve = ram_reserve if memory.unified_memory else 0
    unified_budget = ram_budget if memory.unified_memory else 0

    reasons = [*assets.unresolved, *resolution_reasons]
    actions: tuple[str, ...] = ()
    if projected_vram and memory.available_vram_bytes is None:
        status = ResourceAdmissionStatus.UNAVAILABLE
        reasons.insert(
            0,
            memory.detail or "Selected accelerator memory could not be measured.",
        )
    elif (
        memory.unified_memory
        and projected_vram > int(memory.available_vram_bytes or 0)
    ):
        status = ResourceAdmissionStatus.BLOCKED
        reasons.append(
            "Projected unified accelerator residency exceeds the available "
            "recommended Metal working set."
        )
    elif memory.unified_memory and projected_unified > memory.available_ram_bytes:
        status = ResourceAdmissionStatus.BLOCKED
        reasons.append(
            "Projected unified-memory residency exceeds currently available "
            "system memory."
        )
    elif (
        not memory.unified_memory
        and projected_vram > int(memory.available_vram_bytes or 0)
    ):
        status = ResourceAdmissionStatus.BLOCKED
        reasons.append("Projected CUDA residency exceeds currently available VRAM.")
    elif not memory.unified_memory and projected_ram > memory.available_ram_bytes:
        status = ResourceAdmissionStatus.BLOCKED
        reasons.append("Projected host residency exceeds currently available RAM.")
    elif (
        memory.unified_memory
        and projected_vram
        and projected_vram + vram_reserve
        > int(memory.available_vram_bytes or 0)
    ):
        status = ResourceAdmissionStatus.RISK
        reasons.append(
            "Projected unified accelerator residency would consume the "
            "recommended Metal working-set reserve."
        )
    elif (
        memory.unified_memory
        and projected_unified + unified_reserve > memory.available_ram_bytes
    ):
        status = ResourceAdmissionStatus.RISK
        reasons.append(
            "Projected unified-memory residency would consume the system-memory "
            "safety reserve."
        )
    elif (
        not memory.unified_memory
        and projected_vram
        and projected_vram + vram_reserve
        > int(memory.available_vram_bytes or 0)
    ):
        status = ResourceAdmissionStatus.RISK
        reasons.append("Projected CUDA residency would consume the VRAM safety reserve.")
    elif (
        not memory.unified_memory
        and projected_ram + ram_reserve > memory.available_ram_bytes
    ):
        status = ResourceAdmissionStatus.RISK
        reasons.append("Projected host residency would consume the safety reserve.")
    elif reasons:
        status = ResourceAdmissionStatus.UNAVAILABLE
    else:
        status = ResourceAdmissionStatus.SAFE

    if status is not ResourceAdmissionStatus.SAFE:
        if any(
            "host residency" in reason or "system-memory" in reason
            for reason in reasons
        ):
            actions = (
                "Close other memory-heavy applications, then check again.",
                "Reduce context or prompt batch without changing the model.",
                "Choose a remote provider or a model that fits available system memory.",
            )
        elif any("Multiple visible GPUs" in reason for reason in reasons):
            actions = (
                "Select one GPU with CUDA_VISIBLE_DEVICES, then check again.",
            )
        elif status is ResourceAdmissionStatus.UNAVAILABLE:
            actions = (
                "Resolve the unavailable model, endpoint, shard, or metadata fact, "
                "then re-test the provider and check memory again.",
            )
        elif memory.unified_memory:
            actions = (
                "Close other memory- or accelerator-heavy applications, then check again.",
                "Reduce context or prompt batch without changing the model.",
                "Disable acceleration for a CPU-only run or choose a remote provider.",
            )
        else:
            actions = (
                "Use Automatic or reduce explicit GPU layers, context, or prompt batch.",
                "Close other GPU applications, then run the memory check again.",
                "Choose a provider and model combination that fits the measured budget.",
            )

    return RuntimeResourceAdmissionReport(
        status=status,
        settings_fingerprint=normalized_fingerprint,
        pipeline_values_fingerprint=canonical_fingerprint(values),
        effective_pipeline_values_fingerprint=canonical_fingerprint(effective_values),
        checked_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        gpu_name=memory.gpu_name,
        total_vram_bytes=memory.total_vram_bytes,
        available_vram_bytes=memory.available_vram_bytes,
        projected_vram_bytes=projected_vram,
        vram_reserve_bytes=vram_reserve,
        vram_budget_bytes=vram_budget,
        total_ram_bytes=memory.total_ram_bytes,
        available_ram_bytes=memory.available_ram_bytes,
        projected_ram_bytes=projected_ram,
        ram_reserve_bytes=ram_reserve,
        ram_budget_bytes=ram_budget,
        components=tuple(components),
        reasons=tuple(dict.fromkeys(reason for reason in reasons if str(reason).strip())),
        actions=actions,
        runtime_overrides=runtime_overrides,
        warnings=tuple(dict.fromkeys(assets.warnings)),
        recommended_changes=(),
        backend=memory.backend,
        onnx_backend=memory.onnx_backend,
        llama_server_backend=memory.llama_server_backend,
        llama_cpp_backend=memory.llama_cpp_backend,
        unified_memory=memory.unified_memory,
        projected_unified_bytes=projected_unified,
        unified_reserve_bytes=unified_reserve,
        unified_budget_bytes=unified_budget,
    )


class RuntimeResourceAdmissionService:
    """Probe the host and assess one compiled run without loading a model."""

    def __init__(
        self,
        *,
        memory_probe: Callable[[], RuntimeMemorySnapshot] | None = None,
        asset_probe: Callable[[Mapping[str, object]], RuntimeResourceAssets] = (
            lambda values: probe_runtime_assets(values)
        ),
        compute: ComputeCapabilitySnapshot | None = None,
    ) -> None:
        self._memory_probe = memory_probe or (
            lambda: probe_runtime_memory(compute=compute)
        )
        self._asset_probe = asset_probe

    def evaluate(
        self,
        *,
        settings_fingerprint: str,
        pipeline_values: Mapping[str, object],
    ) -> RuntimeResourceAdmissionReport:
        values = dict(pipeline_values)
        try:
            memory = self._memory_probe()
            assets = self._asset_probe(values)
        except Exception as exc:
            return _unavailable_report(
                settings_fingerprint=settings_fingerprint,
                pipeline_values=values,
                detail=(
                    "Local memory inspection failed "
                    f"({type(exc).__name__})."
                ),
            )
        return assess_runtime_resources(
            settings_fingerprint=settings_fingerprint,
            pipeline_values=values,
            memory=memory,
            assets=assets,
        )


def probe_runtime_memory(
    *,
    compute: ComputeCapabilitySnapshot | None = None,
) -> RuntimeMemorySnapshot:
    total_ram, available_ram = _system_memory()
    devices = _nvidia_memory()
    torch_selection = compute.torch if compute is not None else select_torch_device(True)
    onnx_selection = compute.onnx if compute is not None else select_onnx_providers(True)
    llama_server_backend = (
        compute.llama_server_backend
        if compute is not None
        else probe_llama_server_backend(resolve_llama_server_executable())
    )
    llama_cpp_backend = probe_llama_cpp_python_backend(
        cuda_available=bool(devices),
    )
    if devices:
        primary = devices[0]
        return RuntimeMemorySnapshot(
            total_ram_bytes=total_ram,
            available_ram_bytes=available_ram,
            gpu_name=primary.name,
            total_vram_bytes=primary.total_bytes,
            available_vram_bytes=primary.available_bytes,
            gpu_devices=devices,
            source="psutil+nvidia-smi",
            detail=(
                "Discrete NVIDIA memory was measured; selected runtime "
                "capabilities determine whether it is used."
            ),
            backend=torch_selection.backend,
            onnx_backend=(
                ComputeBackend.CUDA
                if onnx_selection.backend is ComputeBackend.CUDA
                else ComputeBackend.CPU
            ),
            llama_server_backend=llama_server_backend,
            llama_cpp_backend=llama_cpp_backend,
            unified_memory=False,
        )

    mps = probe_mps_memory()
    if mps is not None and mps.recommended_max_bytes > 0 and (
        torch_selection.backend is ComputeBackend.MPS
        or onnx_selection.backend is ComputeBackend.COREML
        or llama_server_backend is ComputeBackend.METAL
        or llama_cpp_backend is ComputeBackend.METAL
    ):
        return RuntimeMemorySnapshot(
            total_ram_bytes=total_ram,
            available_ram_bytes=available_ram,
            gpu_name="Apple Metal",
            total_vram_bytes=mps.recommended_max_bytes,
            available_vram_bytes=mps.available_bytes,
            source="psutil+torch.mps",
            detail=(
                "MPS, CoreML, and Metal share system memory and the recommended "
                "Metal working set."
            ),
            backend=(
                ComputeBackend.MPS
                if torch_selection.backend is ComputeBackend.MPS
                else ComputeBackend.CPU
            ),
            onnx_backend=(
                ComputeBackend.COREML
                if onnx_selection.backend is ComputeBackend.COREML
                else ComputeBackend.CPU
            ),
            llama_server_backend=llama_server_backend,
            llama_cpp_backend=llama_cpp_backend,
            unified_memory=True,
        )

    return RuntimeMemorySnapshot(
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        source="system_memory",
        detail="No measurable local accelerator was selected; using system memory.",
        backend=ComputeBackend.CPU,
        onnx_backend=(
            ComputeBackend.COREML
            if onnx_selection.backend is ComputeBackend.COREML
            else ComputeBackend.CPU
        ),
        llama_server_backend=llama_server_backend,
        llama_cpp_backend=llama_cpp_backend,
        unified_memory=False,
    )


def probe_runtime_assets(pipeline_values: Mapping[str, object]) -> RuntimeResourceAssets:
    values = dict(pipeline_values)
    backend = str(values.get("translator_backend") or "").strip()
    unresolved: list[str] = []
    translation_label = ""
    translation_bytes = 0
    translation_blocks: int | None = None
    translation_shards = 1
    translation_experts = 0
    translation_kv_bytes_per_token = 0
    translation_resident = False
    translation_resident_model_bytes = 0
    translation_resident_vram_bytes = 0
    translation_remote = False
    translation_context_bytes = 0
    translation_vision_workspace_bytes = 0
    translation_incremental_gpu_bytes: int | None = None
    translation_incremental_ram_bytes: int | None = None
    translation_residency_method = ""
    warnings: list[str] = []
    if backend == "GGUF":
        path = Path(str(values.get("gguf_model_path") or "")).expanduser()
        translation_label = path.name or "GGUF translation model"
        shard_paths, shard_error = _resolve_gguf_shards(path)
        if shard_error:
            unresolved.append(shard_error)
        translation_shards = max(1, len(shard_paths))
        translation_bytes = sum(_path_size(item) for item in shard_paths)
        if not translation_bytes or not shard_paths:
            unresolved.append("The selected GGUF model size could not be measured.")
        metadata = tuple(_read_gguf_metadata(item) for item in shard_paths)
        first = metadata[0] if metadata else GgufMetadata()
        translation_blocks = first.block_count
        translation_experts = first.expert_count
        translation_kv_bytes_per_token = first.kv_bytes_per_token
        if metadata and any(
            item.architecture != first.architecture
            or item.block_count != first.block_count
            or item.expert_count != first.expert_count
            for item in metadata[1:]
        ):
            unresolved.append("The selected GGUF shards have inconsistent metadata.")
        if metadata and first.split_count not in {1, len(shard_paths)}:
            unresolved.append("The selected GGUF shard set is incomplete.")
        if first.expert_count > 0 and int(values.get("gguf_n_gpu_layers", -1) or 0) != 0:
            unresolved.append(
                "This MoE GGUF requires CPU layers (0); GPU-layer residency is "
                "not safely measurable from portable metadata."
            )
        if len(shard_paths) > 1:
            translation_label = f"{path.name} · {len(shard_paths)} shards"
    elif backend == "Ollama":
        model = str(values.get("ollama_model") or "").strip()
        translation_label = model or "Ollama translation model"
        if not model or model == "auto-detect":
            unresolved.append(
                "Select an explicit Ollama model before Start so its memory can be budgeted."
            )
        else:
            inventory = _inspect_ollama_runtime(
                endpoint=str(
                    values.get("ollama_base_url") or "http://localhost:11434"
                ),
                model=model,
                configured_context_tokens=max(
                    1, int(values.get("ollama_context", 4096) or 4096)
                ),
            )
            translation_bytes = inventory.model_bytes
            translation_resident = inventory.resident
            translation_resident_model_bytes = inventory.resident_model_bytes
            translation_resident_vram_bytes = inventory.resident_vram_bytes
            translation_remote = inventory.remote
            translation_context_bytes = inventory.context_bytes
            translation_vision_workspace_bytes = inventory.vision_workspace_bytes
            translation_residency_method = (
                "remote_server_owned"
                if inventory.remote
                else (
                    "ollama_resident_capacity_already_reflected:"
                    f"gpu={inventory.resident_vram_bytes}:"
                    f"total={inventory.resident_model_bytes}"
                )
                if inventory.resident
                else "ollama_endpoint_inventory_and_auto_split"
            )
            if inventory.resident:
                translation_incremental_gpu_bytes = 0
                translation_incremental_ram_bytes = 0
            unresolved.extend(inventory.unresolved)
            warnings.extend(inventory.warnings)

    if bool(values.get("use_ollama_discovery", False)):
        discovery_backend = str(values.get("discovery_backend") or "").strip()
        discovery_model = str(values.get("discovery_model") or "").strip()
        same_model = bool(
            discovery_model
            and (
                (
                    backend == "GGUF"
                    and discovery_backend == "GGUF"
                    and os.path.normcase(os.path.abspath(discovery_model))
                    == os.path.normcase(
                        os.path.abspath(str(values.get("gguf_model_path") or ""))
                    )
                )
                or (
                    backend == "Ollama"
                    and discovery_backend == "Ollama"
                    and discovery_model == str(values.get("ollama_model") or "")
                    and str(
                        values.get("discovery_base_url")
                        or "http://localhost:11434"
                    ).rstrip("/")
                    == str(
                        values.get("ollama_base_url")
                        or "http://localhost:11434"
                    ).rstrip("/")
                )
            )
        )
        if not discovery_model:
            unresolved.append(
                "Select an explicit discovery model before enabling LLM discovery."
            )
        elif not same_model:
            unresolved.append(
                "A separate LLM discovery model cannot share this admitted model "
                "budget. Use the selected translation model or disable LLM discovery."
            )

    model_root = Path(models_root())
    detector_root = model_root / "comic-text-detector"
    detector_pt = detector_root / "comictextdetector.pt"
    detector_onnx = detector_root / "comictextdetector.pt.onnx"
    detector_selection = select_torch_device(bool(values.get("use_gpu", True)))
    detector = (
        detector_pt
        if detector_selection.backend is ComputeBackend.CUDA
        else detector_onnx if detector_onnx.is_file() else detector_pt
    )
    manga_ocr_dir = resolve_manga_ocr_system_ref() or resolve_manga_ocr_local_dir(
        str(model_root)
    )
    bubble_bytes = sum(
        _path_size(item)
        for item in (
            resolve_kitsumed_speech_bubble_model(str(model_root)),
            resolve_ogkalu_text_bubble_model(str(model_root)),
        )
    )
    manga_ocr_bytes = _directory_size(manga_ocr_dir)
    if str(values.get("ocr_engine") or "") == "MangaOCR" and not manga_ocr_bytes:
        unresolved.append("The selected MangaOCR model size could not be measured.")
    return RuntimeResourceAssets(
        translation_model_label=translation_label,
        translation_model_bytes=translation_bytes,
        translation_block_count=translation_blocks,
        translation_shard_count=translation_shards,
        translation_expert_count=translation_experts,
        translation_kv_bytes_per_token=translation_kv_bytes_per_token,
        translation_model_already_resident=translation_resident,
        translation_resident_model_bytes=translation_resident_model_bytes,
        translation_resident_vram_bytes=translation_resident_vram_bytes,
        translation_remote=translation_remote,
        translation_context_bytes=translation_context_bytes,
        translation_vision_workspace_bytes=translation_vision_workspace_bytes,
        translation_incremental_gpu_bytes=translation_incremental_gpu_bytes,
        translation_incremental_ram_bytes=translation_incremental_ram_bytes,
        translation_residency_method=translation_residency_method,
        paddle_model_bytes=_path_size(resolve_paddle_ocr_vl_model_file()),
        paddle_mmproj_bytes=_path_size(resolve_paddle_ocr_vl_mmproj_file()),
        manga_ocr_model_bytes=manga_ocr_bytes,
        ner_model_bytes=(420 * MIB if bool(values.get("prescan_use_ner", False)) else 0),
        detector_model_bytes=_path_size(detector),
        bubble_model_bytes=bubble_bytes,
        cleanup_model_bytes=_path_size(resolve_cleanup_inpaint_model_file()),
        font_model_bytes=_path_size(resolve_yuzumarker_font_onnx_file()),
        unresolved=tuple(unresolved),
        warnings=tuple(warnings),
    )


def _torch_component_backend(
    memory: RuntimeMemorySnapshot,
    *,
    use_gpu: bool,
) -> ComputeBackend:
    if not use_gpu:
        return ComputeBackend.CPU
    if memory.backend in {ComputeBackend.CUDA, ComputeBackend.MPS}:
        return memory.backend
    return ComputeBackend.CPU


def _unified_component_bytes(component: ResourceComponentEstimate) -> int:
    """Return incremental shared-pool residency without mirroring model weights.

    MPS/CoreML/Metal weights and their runtime workspace occupy the same physical
    pool represented by ``gpu_bytes``.  Their file-backed host mapping is
    reclaimable and must not be added again.  Partially offloaded translation
    models are the exception: their GPU and CPU values represent disjoint layer
    sets, so both contribute to unified residency.
    """

    if component.backend is ComputeBackend.CPU or not component.gpu_bytes:
        return component.ram_bytes
    if component.component_id == "translation_model" and component.ram_bytes:
        return component.gpu_bytes + component.ram_bytes
    return max(component.gpu_bytes, component.ram_bytes)


def _onnx_component_backend(
    memory: RuntimeMemorySnapshot,
    *,
    use_gpu: bool,
) -> ComputeBackend:
    if not use_gpu:
        return ComputeBackend.CPU
    if memory.onnx_backend in {ComputeBackend.CUDA, ComputeBackend.COREML}:
        return memory.onnx_backend
    return ComputeBackend.CPU


def _detector_component_backend(
    memory: RuntimeMemorySnapshot,
    *,
    use_gpu: bool,
) -> ComputeBackend:
    # ComicTextDetector's portable ONNX adapter is OpenCV DNN, not ORT. Only
    # the Torch `.pt` path has a validated accelerator implementation (CUDA).
    if use_gpu and memory.backend is ComputeBackend.CUDA:
        return ComputeBackend.CUDA
    return ComputeBackend.CPU


def _paddle_component_backend(
    memory: RuntimeMemorySnapshot,
    *,
    use_gpu: bool,
) -> ComputeBackend:
    if not use_gpu:
        return ComputeBackend.CPU
    if memory.llama_server_backend in {ComputeBackend.CUDA, ComputeBackend.METAL}:
        return memory.llama_server_backend
    return ComputeBackend.CPU


def _translation_component_backend(
    memory: RuntimeMemorySnapshot,
    *,
    use_gpu: bool,
    translator_backend: str,
    assets: RuntimeResourceAssets,
) -> ComputeBackend:
    if not use_gpu:
        return ComputeBackend.CPU
    if translator_backend == "GGUF":
        if memory.llama_cpp_backend in {
            ComputeBackend.CUDA,
            ComputeBackend.METAL,
        }:
            return memory.llama_cpp_backend
        return ComputeBackend.CPU
    if translator_backend == "Ollama" and (
        assets.translation_resident_vram_bytes > 0
        or int(assets.translation_incremental_gpu_bytes or 0) > 0
    ):
        if memory.unified_memory:
            return ComputeBackend.METAL
        if memory.gpu_devices:
            return ComputeBackend.CUDA
    return ComputeBackend.CPU


def _append_fixed_component(
    components: list[ResourceComponentEstimate],
    *,
    component_id: str,
    label: str,
    file_bytes: int,
    backend: ComputeBackend,
    gpu_multiplier: float,
) -> None:
    if not file_bytes:
        return
    components.append(
        ResourceComponentEstimate(
            component_id=component_id,
            label=label,
            gpu_bytes=(
                int(round(file_bytes * gpu_multiplier))
                if backend is not ComputeBackend.CPU
                else 0
            ),
            ram_bytes=int(file_bytes),
            method="measured_asset_size_with_runtime_workspace_factor",
            backend=backend,
        )
    )


def _offload_fraction(layers: int, block_count: int | None) -> float:
    if layers == 0:
        return 0.0
    if layers < 0:
        return 1.0
    if block_count is None:
        # Without metadata, preserve the user's setting but use a conservative
        # upper bound rather than silently claiming the partial plan is safe.
        return min(1.0, max(0.25, layers / 64.0))
    return min(1.0, layers / float(max(1, block_count)))


def _auto_fit_gguf_layers(
    *,
    model_bytes: int,
    block_count: int,
    context_tokens: int,
    kv_bytes_per_token: int,
    fixed_gpu_bytes: int,
    memory: RuntimeMemorySnapshot,
) -> int:
    """Resolve Automatic to the highest layer count inside the live reserve.

    Model weights and context stay unchanged.  Zero is a valid fail-safe plan:
    llama.cpp then keeps the translator on the CPU while the other configured
    GPU runtimes retain their measured budget.
    """

    if model_bytes <= 0 or block_count <= 0 or context_tokens <= 0:
        raise ValueError("valid GGUF model, block count, and context are required")
    if memory.available_vram_bytes is None or memory.total_vram_bytes is None:
        return 0
    reserve = max(1536 * MIB, int(memory.total_vram_bytes * 0.10))
    available = max(
        0,
        int(memory.available_vram_bytes) - reserve - max(0, int(fixed_gpu_bytes)),
    )
    kv_bytes = context_tokens * max(
        64 * 1024,
        int(kv_bytes_per_token or block_count * 4096),
    )
    if available <= kv_bytes:
        return 0
    candidate = min(
        block_count,
        max(0, int(block_count * ((available - kv_bytes) / model_bytes))),
    )
    while candidate > 0:
        weights = int(round(model_bytes * (candidate / float(block_count))))
        if weights + kv_bytes <= available:
            return candidate
        candidate -= 1
    return 0


def _paddle_gpu_layers() -> int:
    raw = os.environ.get("MT_PADDLEOCR_VL_N_GPU_LAYERS")
    if raw is None:
        return 99
    try:
        return int(raw)
    except ValueError:
        return 99


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default)) or default))
    except ValueError:
        return default


def _system_memory() -> tuple[int, int]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception as exc:  # pragma: no cover - psutil is pinned in the app env
        raise RuntimeError("System memory could not be measured.") from exc


def _nvidia_memory() -> tuple[RuntimeGpuMemoryDevice, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in {"", "-1"}:
        return ()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creationflags,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ()
    rows = tuple(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
    if not rows:
        return ()
    requested = tuple(
        item.strip()
        for item in str(visible or "").split(",")
        if item.strip()
    )
    selected = tuple(
        row
        for row in rows
        if len(row) >= 5
        and (
            not requested
            or row[0].strip() in requested
            or row[1].strip() in requested
        )
    )
    if requested and not selected:
        return ()
    devices: list[RuntimeGpuMemoryDevice] = []
    try:
        for row in selected or rows:
            devices.append(
                RuntimeGpuMemoryDevice(
                    index=row[0].strip(),
                    uuid=row[1].strip(),
                    name=row[2].strip(),
                    total_bytes=int(float(row[3])) * MIB,
                    available_bytes=int(float(row[4])) * MIB,
                )
            )
    except (IndexError, ValueError):
        return ()
    return tuple(devices)


def _normalize_ollama_endpoint(endpoint: str) -> str:
    value = str(endpoint or "").strip().rstrip("/")
    parsed = urllib_parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Ollama endpoint must be a public HTTP(S) base URL")
    return value


def _ollama_endpoint_is_remote(endpoint: str) -> bool:
    host = (urllib_parse.urlsplit(endpoint).hostname or "").casefold()
    return host not in {"localhost", "127.0.0.1", "::1"}


def _ollama_json_request(
    endpoint: str,
    path: str,
    payload: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    body = json.dumps(dict(payload)).encode("utf-8") if payload is not None else None
    request = urllib_request.Request(
        f"{endpoint}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib_request.urlopen(request, timeout=5) as response:
            raw = response.read(16 * MIB + 1)
    except (OSError, urllib_error.URLError, urllib_error.HTTPError) as exc:
        raise RuntimeError(f"Ollama endpoint inspection failed for {path}") from exc
    if len(raw) > 16 * MIB:
        raise RuntimeError("Ollama endpoint response is too large")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Ollama endpoint returned invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("Ollama endpoint returned a non-object payload")
    return value


def _ollama_model_entry(
    payload: Mapping[str, object],
    model: str,
) -> Mapping[str, object] | None:
    rows = payload.get("models")
    if not isinstance(rows, list):
        return None
    exact: list[Mapping[str, object]] = []
    compatible: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or row.get("model") or "").strip()
        if name == model:
            exact.append(row)
        elif ":" not in model and name == f"{model}:latest":
            compatible.append(row)
    matches = exact or compatible
    return matches[0] if len(matches) == 1 else None


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _ollama_context_residency_bytes(
    show: Mapping[str, object],
    context_tokens: int,
) -> int:
    info = show.get("model_info")
    if not isinstance(info, Mapping):
        return 0
    architecture = str(info.get("general.architecture") or "").strip()
    if not architecture:
        return 0
    blocks = _nonnegative_int(info.get(f"{architecture}.block_count"))
    embedding = _nonnegative_int(info.get(f"{architecture}.embedding_length"))
    heads = _nonnegative_int(info.get(f"{architecture}.attention.head_count"))
    kv_heads = _nonnegative_int(
        info.get(f"{architecture}.attention.head_count_kv")
    ) or heads
    if not blocks or not embedding or not heads or not kv_heads:
        return 0
    head_width = max(1, embedding // heads)
    key_width = _nonnegative_int(
        info.get(f"{architecture}.attention.key_length")
    ) or head_width
    value_width = _nonnegative_int(
        info.get(f"{architecture}.attention.value_length")
    ) or head_width
    # Ollama's portable API does not expose KV quantization.  FP16 is the
    # conservative supported default for a provider-independent estimate.
    return max(1, context_tokens) * blocks * kv_heads * (key_width + value_width) * 2


def _inspect_ollama_runtime(
    *,
    endpoint: str,
    model: str,
    configured_context_tokens: int,
    request_json: Callable[
        [str, str, Mapping[str, object] | None], Mapping[str, object]
    ] = _ollama_json_request,
) -> OllamaRuntimeInventory:
    normalized_model = str(model or "").strip()
    if not normalized_model or normalized_model == "auto-detect":
        return OllamaRuntimeInventory(
            endpoint=str(endpoint or ""),
            model=normalized_model,
            configured_context_tokens=max(1, configured_context_tokens),
            unresolved=("Select an explicit Ollama model before Start.",),
        )
    try:
        normalized_endpoint = _normalize_ollama_endpoint(endpoint)
    except ValueError as exc:
        return OllamaRuntimeInventory(
            endpoint=str(endpoint or ""),
            model=normalized_model,
            configured_context_tokens=max(1, configured_context_tokens),
            unresolved=(str(exc),),
        )
    remote = _ollama_endpoint_is_remote(normalized_endpoint)
    unresolved: list[str] = []
    warnings: list[str] = []
    try:
        tags = request_json(normalized_endpoint, "/api/tags", None)
    except RuntimeError as exc:
        return OllamaRuntimeInventory(
            endpoint=normalized_endpoint,
            model=normalized_model,
            configured_context_tokens=max(1, configured_context_tokens),
            remote=remote,
            unresolved=(str(exc),),
        )
    tag = _ollama_model_entry(tags, normalized_model)
    if tag is None:
        return OllamaRuntimeInventory(
            endpoint=normalized_endpoint,
            model=normalized_model,
            configured_context_tokens=max(1, configured_context_tokens),
            remote=remote,
            unresolved=("The configured Ollama model was not found at this endpoint.",),
        )
    model_bytes = _nonnegative_int(tag.get("size"))
    if not model_bytes:
        unresolved.append("The configured Ollama model size could not be measured.")
    show: Mapping[str, object] = {}
    try:
        show = request_json(
            normalized_endpoint,
            "/api/show",
            {"model": normalized_model},
        )
    except RuntimeError:
        if remote:
            warnings.append(
                "Remote Ollama model metadata is unavailable; remote capacity "
                "remains server-owned."
            )
        else:
            unresolved.append("Ollama model metadata could not be inspected.")
    ps: Mapping[str, object] = {}
    try:
        ps = request_json(normalized_endpoint, "/api/ps", None)
    except RuntimeError:
        warnings.append(
            "Ollama residency could not be inspected; the model is budgeted as "
            "not resident."
        )
    resident_entry = _ollama_model_entry(ps, normalized_model)
    resident = resident_entry is not None
    resident_model_bytes = (
        _nonnegative_int(resident_entry.get("size")) if resident_entry else 0
    )
    resident_vram_bytes = (
        _nonnegative_int(resident_entry.get("size_vram"))
        if resident_entry
        else 0
    )
    resident_context = (
        _nonnegative_int(resident_entry.get("context_length"))
        if resident_entry
        else 0
    )
    configured_context = max(1, int(configured_context_tokens))
    configured_context_bytes = _ollama_context_residency_bytes(
        show, configured_context
    )
    resident_context_bytes = _ollama_context_residency_bytes(
        show, resident_context
    )
    if not remote and not configured_context_bytes:
        unresolved.append("Ollama context/KV residency could not be measured.")
    context_bytes = (
        max(0, configured_context_bytes - resident_context_bytes)
        if resident
        else configured_context_bytes
    )
    capabilities = show.get("capabilities")
    vision = bool(
        isinstance(capabilities, list)
        and any(str(item).casefold() == "vision" for item in capabilities)
    )
    vision_workspace = 0
    if vision and not resident:
        projector_info = show.get("projector_info")
        projector_parameters = (
            _nonnegative_int(projector_info.get("general.parameter_count"))
            if isinstance(projector_info, Mapping)
            else 0
        )
        if projector_parameters:
            vision_workspace = max(512 * MIB, projector_parameters * 2)
        elif remote:
            warnings.append(
                "Remote Ollama vision-projector capacity is not exposed; it "
                "remains server-owned."
            )
        else:
            unresolved.append(
                "Ollama vision-projector residency could not be measured."
            )
    if remote:
        warnings.append(
            "Remote Ollama capacity is server-owned; this check covers local "
            "OCR, detection, cleanup, and rendering resources only."
        )
        context_bytes = 0
        vision_workspace = 0
    return OllamaRuntimeInventory(
        endpoint=normalized_endpoint,
        model=normalized_model,
        model_bytes=model_bytes,
        resident=resident,
        resident_model_bytes=resident_model_bytes,
        resident_vram_bytes=resident_vram_bytes,
        configured_context_tokens=configured_context,
        resident_context_tokens=resident_context,
        context_bytes=context_bytes,
        vision_workspace_bytes=vision_workspace,
        remote=remote,
        unresolved=tuple(dict.fromkeys(unresolved)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _parse_size(value: str, unit: str) -> int:
    try:
        number = float(value)
    except ValueError:
        return 0
    factor = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
    }.get(str(unit).upper(), 0)
    return int(number * factor)


def _path_size(path: str | os.PathLike[str] | None) -> int:
    try:
        candidate = Path(path) if path else None
        return int(candidate.stat().st_size) if candidate and candidate.is_file() else 0
    except OSError:
        return 0


def _directory_size(path: str | os.PathLike[str] | None) -> int:
    try:
        candidate = Path(path) if path else None
        if candidate is None or not candidate.is_dir():
            return 0
        return sum(
            item.stat().st_size
            for item in candidate.rglob("*")
            if item.is_file()
        )
    except OSError:
        return 0


_GGUF_SHARD_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)


def _resolve_gguf_shards(path: Path) -> tuple[tuple[Path, ...], str]:
    if not path.is_file():
        return (), "The selected GGUF model file does not exist."
    match = _GGUF_SHARD_PATTERN.match(path.name)
    if match is None:
        return (path,), ""
    count = int(match.group("count"))
    if count <= 0 or count > 10_000:
        return (), "The selected GGUF shard count is invalid."
    prefix = match.group("prefix")
    candidates: dict[int, Path] = {}
    for candidate in path.parent.iterdir():
        item = _GGUF_SHARD_PATTERN.match(candidate.name)
        if (
            item is None
            or item.group("prefix").casefold() != prefix.casefold()
            or int(item.group("count")) != count
        ):
            continue
        index = int(item.group("index"))
        if index in candidates:
            return (), "The selected GGUF shard set contains duplicate indexes."
        candidates[index] = candidate
    expected = set(range(1, count + 1))
    if set(candidates) != expected:
        return (), "The selected GGUF shard set is incomplete."
    return tuple(candidates[index] for index in sorted(candidates)), ""


def _read_gguf_metadata(path: Path) -> GgufMetadata:
    if not path.is_file():
        return GgufMetadata()
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def exact(stream, count: int) -> bytes:
        value = stream.read(count)
        if len(value) != count:
            raise ValueError("unexpected GGUF EOF")
        return value

    def u32(stream) -> int:
        return struct.unpack("<I", exact(stream, 4))[0]

    def u64(stream) -> int:
        return struct.unpack("<Q", exact(stream, 8))[0]

    def string(stream) -> str:
        length = u64(stream)
        if length > 16 * MIB:
            raise ValueError("GGUF metadata string is too large")
        return exact(stream, length).decode("utf-8", errors="replace")

    def scalar(stream, value_type: int):
        size = scalar_sizes.get(value_type)
        if size is None:
            raise ValueError("unsupported GGUF scalar type")
        raw = exact(stream, size)
        formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        return struct.unpack(formats[value_type], raw)[0]

    def value(stream, value_type: int):
        if value_type == 8:
            return string(stream)
        if value_type == 9:
            element_type = u32(stream)
            count = u64(stream)
            if count > 10_000_000:
                raise ValueError("GGUF metadata array is too large")
            if element_type == 8:
                for _ in range(count):
                    string(stream)
            else:
                size = scalar_sizes.get(element_type)
                if size is None:
                    raise ValueError("unsupported GGUF array type")
                exact(stream, size * count)
            return None
        return scalar(stream, value_type)

    try:
        with path.open("rb") as stream:
            if exact(stream, 4) != b"GGUF" or u32(stream) not in (2, 3):
                return GgufMetadata()
            tensor_count = u64(stream)
            metadata_count = u64(stream)
            if metadata_count > 1_000_000:
                return GgufMetadata()
            architecture = ""
            block_counts: dict[str, int] = {}
            expert_counts: dict[str, int] = {}
            embedding_lengths: dict[str, int] = {}
            head_counts: dict[str, int] = {}
            kv_head_counts: dict[str, int] = {}
            key_lengths: dict[str, int] = {}
            value_lengths: dict[str, int] = {}
            split_count = 1
            split_index = 0
            for _ in range(metadata_count):
                key = string(stream)
                item = value(stream, u32(stream))
                if key == "general.architecture" and isinstance(item, str):
                    architecture = item
                elif key.endswith(".block_count") and isinstance(item, int):
                    block_counts[key[: -len(".block_count")]] = int(item)
                elif key.endswith(".expert_count") and isinstance(item, int):
                    expert_counts[key[: -len(".expert_count")]] = int(item)
                elif key.endswith(".embedding_length") and isinstance(item, int):
                    embedding_lengths[key[: -len(".embedding_length")]] = int(item)
                elif key.endswith(".attention.head_count") and isinstance(item, int):
                    head_counts[key[: -len(".attention.head_count")]] = int(item)
                elif key.endswith(".attention.head_count_kv") and isinstance(item, int):
                    kv_head_counts[key[: -len(".attention.head_count_kv")]] = int(item)
                elif key.endswith(".attention.key_length") and isinstance(item, int):
                    key_lengths[key[: -len(".attention.key_length")]] = int(item)
                elif key.endswith(".attention.value_length") and isinstance(item, int):
                    value_lengths[key[: -len(".attention.value_length")]] = int(item)
                elif key == "split.count" and isinstance(item, int):
                    split_count = int(item)
                elif key == "split.no" and isinstance(item, int):
                    split_index = int(item)
            blocks = block_counts.get(architecture)
            embedding = max(0, embedding_lengths.get(architecture, 0))
            heads = max(0, head_counts.get(architecture, 0))
            kv_heads = max(0, kv_head_counts.get(architecture, 0)) or heads
            head_width = embedding // heads if embedding and heads else 0
            key_width = max(0, key_lengths.get(architecture, 0)) or head_width
            value_width = max(0, value_lengths.get(architecture, 0)) or head_width
            kv_bytes_per_token = (
                int(blocks) * kv_heads * (key_width + value_width) * 2
                if blocks and kv_heads and key_width and value_width
                else 0
            )
            return GgufMetadata(
                architecture=architecture,
                block_count=blocks,
                expert_count=max(0, int(expert_counts.get(architecture, 0))),
                split_count=max(1, split_count),
                split_index=max(0, split_index),
                tensor_count=max(0, int(tensor_count)),
                embedding_length=embedding,
                kv_bytes_per_token=kv_bytes_per_token,
            )
    except (OSError, ValueError, struct.error):
        return GgufMetadata()


def _format_bytes(value: int) -> str:
    amount = max(0, int(value))
    if amount >= GIB:
        return f"{amount / GIB:.1f} GiB"
    return f"{amount / MIB:.0f} MiB"


def _unavailable_report(
    *,
    settings_fingerprint: str,
    pipeline_values: Mapping[str, object],
    detail: str,
) -> RuntimeResourceAdmissionReport:
    return RuntimeResourceAdmissionReport(
        status=ResourceAdmissionStatus.UNAVAILABLE,
        settings_fingerprint=str(settings_fingerprint or "unavailable"),
        pipeline_values_fingerprint=canonical_fingerprint(dict(pipeline_values)),
        effective_pipeline_values_fingerprint=canonical_fingerprint(
            dict(pipeline_values)
        ),
        checked_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        gpu_name="",
        total_vram_bytes=None,
        available_vram_bytes=None,
        projected_vram_bytes=0,
        vram_reserve_bytes=0,
        vram_budget_bytes=0,
        total_ram_bytes=1,
        available_ram_bytes=0,
        projected_ram_bytes=0,
        ram_reserve_bytes=0,
        ram_budget_bytes=0,
        components=(),
        reasons=(str(detail or "Local memory could not be measured."),),
        actions=(
            "Verify system-memory and selected-accelerator reporting, then run the "
            "check again.",
        ),
        runtime_overrides=(),
        warnings=(),
        recommended_changes=(),
    )


__all__ = [
    "GIB",
    "ResourceAdmissionStatus",
    "ResourceMonitorLevel",
    "ResourceComponentEstimate",
    "RuntimeGpuMemoryDevice",
    "RuntimeMemorySnapshot",
    "RuntimeResourceAdmissionReport",
    "RuntimeResourceAdmissionService",
    "RuntimeResourceMonitorReport",
    "RuntimeResourceMonitorService",
    "RuntimeResourceAssets",
    "assess_runtime_resources",
    "monitor_runtime_reserve",
    "probe_runtime_assets",
    "probe_runtime_memory",
]
