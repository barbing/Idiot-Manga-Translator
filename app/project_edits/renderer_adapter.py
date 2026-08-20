# -*- coding: utf-8 -*-
"""Pixel-parity adapter from the existing page renderer to isolated layers.

The adapter does not implement layout, fitting, typography, or rasterization.
It invokes the existing ``PageRenderExecutor`` for exactly one parent, then
replays the executor-owned adjusted plan/layout/report through the same
compositor onto a transparent surface.  The layer is accepted only when
recomposing it over the selected CleanedPageBase is pixel-identical to the
executor's authoritative output.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict, is_dataclass
import importlib
import importlib.metadata
import marshal
import os
import platform
import sys
import tempfile
import threading
import time
from typing import Any, Mapping

from app.render.compositor import RENDERER_COMPOSITOR_VERSION
from app.render.font_manager import FONT_MANAGER_VERSION
from app.render.ink_bound_layout_fitter import INK_BOUND_LAYOUT_FITTER_VERSION
from app.render.layout_planner import RENDER_LAYOUT_PLANNER_VERSION
from app.render.render_execution import PAGE_RENDER_EXECUTOR_VERSION, PageRenderExecutor
from app.render.typesetting_contracts import (
    FitReport,
    RenderLayerPlan,
    TypesetLayout,
)

from .fingerprints import canonical_sha256
from .layer_cache import file_sha256

try:
    from PIL import Image, ImageChops
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageChops = None


EFFECTIVE_LAYER_RENDERER_ADAPTER_VERSION = "effective_layer_renderer_adapter_v3"


class EffectiveLayerRendererError(RuntimeError):
    """An existing renderer result cannot form an exact isolated layer."""


@dataclass(frozen=True, slots=True)
class IsolatedLayerRenderResult:
    plan: RenderLayerPlan
    layout: TypesetLayout
    fit_report: FitReport
    authoritative_audit: Mapping[str, Any]
    extraction_audit: Mapping[str, Any]
    surface: Any
    elapsed_ms: float


class EffectiveLayerRendererAdapter:
    """Expose exact isolated parent surfaces without changing renderer logic."""

    def __init__(self, executor: PageRenderExecutor | None = None) -> None:
        self.executor = executor or PageRenderExecutor()
        # The renderer owns mutable font/shaping/raster caches.  One adapter
        # serializes its own jobs so GUI workers cannot race that cache graph.
        self._render_lock = threading.RLock()
        self.renderer_contract_fingerprint = _renderer_contract_fingerprint(
            self.executor
        )
        self.font_asset_fingerprint = _font_asset_inventory_fingerprint(
            self.executor
        )

    def render_isolated_layer(
        self,
        *,
        cleaned_page_base_path: str,
        plan: RenderLayerPlan,
        working_directory: str,
    ) -> IsolatedLayerRenderResult:
        with self._render_lock:
            return self._render_isolated_layer(
                cleaned_page_base_path=cleaned_page_base_path,
                plan=plan,
                working_directory=working_directory,
            )

    def _render_isolated_layer(
        self,
        *,
        cleaned_page_base_path: str,
        plan: RenderLayerPlan,
        working_directory: str,
    ) -> IsolatedLayerRenderResult:
        if Image is None or ImageChops is None:
            raise RuntimeError("Pillow is not installed")
        if not os.path.isfile(cleaned_page_base_path):
            raise EffectiveLayerRendererError("CleanedPageBase asset is missing")
        os.makedirs(working_directory, exist_ok=True)
        handle, authoritative_path = tempfile.mkstemp(
            prefix=".gui-parent-authority-",
            suffix=".png",
            dir=working_directory,
        )
        os.close(handle)
        started = time.perf_counter()
        try:
            render_result = self.executor.compose(
                cleaned_page_base_path,
                authoritative_path,
                [plan],
            )
            if (
                render_result.status != "completed"
                or not render_result.output_committed
                or render_result.failed_layer_ids
                or len(render_result.plans) != 1
                or len(render_result.layouts) != 1
                or len(render_result.fit_reports) != 1
                or len(render_result.layer_audits) != 1
            ):
                reason = str(render_result.failure_reason or "")
                issues = ",".join(str(item) for item in render_result.issues)
                raise EffectiveLayerRendererError(
                    "renderer rejected the effective parent layer"
                    + (f": {reason}" if reason else "")
                    + (f" [{issues}]" if issues else "")
                )

            adjusted_plan = render_result.plans[0]
            layout = render_result.layouts[0]
            report = render_result.fit_reports[0]
            authoritative_audit = dict(render_result.layer_audits[0])
            if adjusted_plan.parent_id != plan.parent_id:
                raise EffectiveLayerRendererError(
                    "renderer changed the effective parent identity"
                )
            _require_protected_plan_fields_unchanged(plan, adjusted_plan)
            has_visible_text = bool(str(adjusted_plan.translated_text or ""))
            if has_visible_text and not bool(authoritative_audit.get("drawn")):
                raise EffectiveLayerRendererError(
                    "renderer did not draw a non-empty effective parent layer"
                )
            if has_visible_text and not bool(
                authoritative_audit.get("text_placement_complete")
            ):
                raise EffectiveLayerRendererError(
                    "renderer did not place the complete effective target text"
                )
            with Image.open(cleaned_page_base_path) as source:
                cleaned_page = source.convert("RGBA")
            transparent = Image.new("RGBA", cleaned_page.size, (0, 0, 0, 0))
            replay_audit = self.executor.compositor.compose_layer(
                transparent,
                adjusted_plan,
                layout,
                report,
            )
            if bool(replay_audit.get("drawn")) != bool(
                authoritative_audit.get("drawn")
            ):
                raise EffectiveLayerRendererError(
                    "isolated layer replay changed renderer draw status"
                )
            if has_visible_text and not bool(replay_audit.get("drawn")):
                raise EffectiveLayerRendererError(
                    "isolated layer replay produced no visible pixels"
                )

            recomposed = cleaned_page.copy()
            recomposed.alpha_composite(transparent)
            with Image.open(authoritative_path) as rendered:
                authoritative_page = rendered.convert("RGBA")
            if authoritative_page.size != recomposed.size:
                raise EffectiveLayerRendererError(
                    "isolated layer replay changed the page dimensions"
                )
            difference = ImageChops.difference(authoritative_page, recomposed)
            if difference.getbbox(alpha_only=False) is not None:
                raise EffectiveLayerRendererError(
                    "isolated layer replay is not pixel-identical to renderer output"
                )
            extraction_audit = {
                "effective_layer_renderer_adapter_version": (
                    EFFECTIVE_LAYER_RENDERER_ADAPTER_VERSION
                ),
                "renderer_contract_fingerprint": self.renderer_contract_fingerprint,
                "font_asset_fingerprint": self.font_asset_fingerprint,
                "requested_plan_fingerprint": canonical_sha256(
                    _plan_payload(plan)
                ),
                "adjusted_plan_fingerprint": canonical_sha256(
                    _plan_payload(adjusted_plan)
                ),
                "authoritative_output_sha256": file_sha256(authoritative_path),
                "pixel_parity": True,
                "pixel_difference_used_for_layer_extraction": False,
                "pixel_difference_used_for_parity_validation": True,
                "same_executor_compositor_used": True,
                "canvas_size": [int(recomposed.width), int(recomposed.height)],
            }
            return IsolatedLayerRenderResult(
                plan=adjusted_plan,
                layout=layout,
                fit_report=report,
                authoritative_audit=authoritative_audit,
                extraction_audit=extraction_audit,
                surface=transparent,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        finally:
            try:
                if os.path.exists(authoritative_path):
                    os.unlink(authoritative_path)
            except OSError:
                pass


def _renderer_contract_fingerprint(executor: PageRenderExecutor) -> str:
    return canonical_sha256(
        {
            "effective_layer_renderer_adapter_version": (
                EFFECTIVE_LAYER_RENDERER_ADAPTER_VERSION
            ),
            "page_render_executor_version": PAGE_RENDER_EXECUTOR_VERSION,
            "layout_planner_version": str(
                getattr(executor.layout_planner, "version", "")
                or RENDER_LAYOUT_PLANNER_VERSION
            ),
            "typesetting_engine_version": str(
                getattr(executor.typesetting_engine, "version", "") or "unversioned"
            ),
            "ink_bound_layout_fitter_version": str(
                getattr(executor.ink_bound_fitter, "version", "")
                or INK_BOUND_LAYOUT_FITTER_VERSION
            ),
            "renderer_compositor_version": RENDERER_COMPOSITOR_VERSION,
            "font_manager_version": FONT_MANAGER_VERSION,
            "glyph_rasterizer_type": type(executor.compositor.glyph_rasterizer).__name__,
            "component_configuration": _renderer_component_configuration(executor),
            "renderer_source_sha256": _renderer_source_fingerprints(),
            "runtime_dependencies": _runtime_dependency_versions(executor),
            "python_runtime": {
                "implementation": platform.python_implementation(),
                "version": list(sys.version_info[:3]),
                "architecture": platform.machine(),
            },
        }
    )


def _renderer_component_configuration(executor: PageRenderExecutor) -> dict[str, Any]:
    typesetter = executor.typesetting_engine
    return {
        "font_manager": _component_identity(executor.font_manager),
        "typesetting_engine": _component_identity(typesetter),
        "typesetting_policy": _stable_configuration(
            getattr(typesetter, "policy", None)
        ),
        "text_shaper": _component_identity(getattr(typesetter, "shaper", None)),
        "line_break_planner": _component_identity(
            getattr(typesetter, "break_planner", None)
        ),
        "lexical_segmenter": _component_identity(
            getattr(typesetter, "lexical_segmenter", None)
        ),
        "lexical_segmenter_assets": _lexical_segmenter_identity(
            getattr(typesetter, "lexical_segmenter", None)
        ),
        "layout_planner": _component_identity(executor.layout_planner),
        "ink_bound_fitter": _component_identity(executor.ink_bound_fitter),
        "compositor": _component_identity(executor.compositor),
        "glyph_rasterizer": _component_identity(
            getattr(executor.compositor, "glyph_rasterizer", None)
        ),
    }


def _component_identity(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "unavailable", "version": ""}
    cls = type(value)
    return {
        "type": f"{cls.__module__}.{cls.__qualname__}",
        "version": str(getattr(value, "version", "") or ""),
    }


def _stable_configuration(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _stable_configuration(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _stable_configuration(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_configuration(item) for item in value]
    return str(value)


def _renderer_source_fingerprints() -> dict[str, str]:
    modules = (
        "app.render.compositor",
        "app.render.font_manager",
        "app.render.glyph_rasterizer",
        "app.render.ink_bound_layout_fitter",
        "app.render.layout_planner",
        "app.render.line_break_planner",
        "app.render.parent_layer_effects",
        "app.render.render_execution",
        "app.render.render_layer_adapter",
        "app.render.source_punctuation_hints",
        "app.render.target_lexical_segmenter",
        "app.render.text_shaper",
        "app.render.typesetting_contracts",
        "app.render.typesetting_engine",
        "app.render.typesetting_text",
        "app.project_edits.renderer_adapter",
    )
    result: dict[str, str] = {}
    for module_name in modules:
        module = importlib.import_module(module_name)
        source_path = os.path.abspath(str(getattr(module, "__file__", "") or ""))
        if source_path.endswith((".pyc", ".pyo")) and os.path.isfile(source_path[:-1]):
            source_path = source_path[:-1]
        if not source_path or not os.path.isfile(source_path):
            loader = getattr(module, "__loader__", None)
            get_code = getattr(loader, "get_code", None)
            code = get_code(module_name) if callable(get_code) else None
            if code is None:
                raise EffectiveLayerRendererError(
                    f"renderer contract source is unavailable: {module_name}"
                )
            result[module_name] = canonical_sha256(
                {
                    "kind": "frozen_loader_code",
                    "marshal_sha256": __import__("hashlib").sha256(
                        marshal.dumps(code)
                    ).hexdigest(),
                }
            )
            continue
        result[module_name] = file_sha256(source_path)
    return result


def _runtime_dependency_versions(executor: PageRenderExecutor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for distribution in (
        "Pillow",
        "freetype-py",
        "uharfbuzz",
        "PyICU",
        "numpy",
        "opencv-python",
        "fonttools",
        "regex",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "unavailable"
    try:
        icu = importlib.import_module("icu")
        result["icu_runtime"] = {
            "pyicu_version": str(getattr(icu, "VERSION", "")),
            "icu_version": str(getattr(icu, "ICU_VERSION", "")),
            "unicode_version": str(getattr(icu, "UNICODE_VERSION", "")),
        }
    except Exception:
        result["icu_runtime"] = {
            "pyicu_version": "unavailable",
            "icu_version": "unavailable",
            "unicode_version": "unavailable",
        }
    result["lexical_segmenter"] = _lexical_segmenter_identity(
        getattr(executor.typesetting_engine, "lexical_segmenter", None)
    )
    try:
        freetype = importlib.import_module("freetype")
        version = getattr(freetype, "version", None)
        result["native_freetype"] = (
            list(version()) if callable(version) else "unknown"
        )
    except Exception:
        result["native_freetype"] = "unavailable"
    try:
        harfbuzz = importlib.import_module("uharfbuzz")
        version_string = getattr(harfbuzz, "version_string", None)
        result["native_harfbuzz"] = (
            str(version_string()) if callable(version_string) else "unknown"
        )
    except Exception:
        result["native_harfbuzz"] = "unavailable"
    try:
        features = importlib.import_module("PIL.features")
        result["pillow_native"] = {
            name: str(features.version(name) or "unavailable")
            for name in ("freetype2", "libjpeg_turbo", "libtiff", "webp")
        }
    except Exception:
        result["pillow_native"] = {"status": "unavailable"}
    return result


def _font_asset_inventory_fingerprint(executor: PageRenderExecutor) -> str:
    records: list[dict[str, Any]] = []
    for face in executor.font_manager.available_faces():
        path = os.path.abspath(str(face.path or ""))
        if not path or not os.path.isfile(path):
            continue
        records.append(
            {
                "face_id": str(face.face_id),
                "path": path,
                "sha256": file_sha256(path),
                "family": str(face.family),
                "style_class": str(face.style_class),
                "weight": str(face.weight),
                "source": str(face.source),
                "serif": bool(face.serif),
                "monospace": bool(face.monospace),
                "priority": int(face.priority),
            }
        )
    if not records:
        raise EffectiveLayerRendererError("registered renderer font assets are unavailable")
    roles = [
        value.to_audit_dict()
        for value in executor.font_manager.required_role_inventory()
    ]
    return canonical_sha256(
        {
            "faces": sorted(records, key=lambda item: item["face_id"]),
            "required_roles": sorted(roles, key=lambda item: item["role_id"]),
        }
    )


def _lexical_segmenter_identity(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "available": False,
            "package_version": "",
            "dictionary_sha256": "",
            "hmm_model_sha256": "",
            "issues": ["unavailable"],
        }
    return {
        "available": bool(getattr(value, "available", False)),
        "package_version": str(getattr(value, "package_version", "") or ""),
        "dictionary_sha256": str(
            getattr(value, "dictionary_sha256", "") or ""
        ),
        "hmm_model_sha256": str(
            getattr(value, "hmm_model_sha256", "") or ""
        ),
        "issues": [str(item) for item in getattr(value, "issues", ())],
    }


def _plan_payload(plan: RenderLayerPlan) -> dict[str, Any]:
    payload = plan.to_audit_dict()
    payload.pop("render_layer_plan_version", None)
    return payload


def _require_protected_plan_fields_unchanged(
    requested: RenderLayerPlan,
    adjusted: RenderLayerPlan,
) -> None:
    requested_payload = _plan_payload(requested)
    adjusted_payload = _plan_payload(adjusted)
    # The existing layout planner owns page-local slot geometry and its
    # derived metadata.  Shape-aware planning also records the selected safe
    # box in ``clipping_region_ref``.  The upstream cleanup/root/allowed-area
    # references remain protected; only the two renderer-owned audit boxes may
    # be added.  User-authoritative text/style/role/base identity and every
    # other renderer input must survive unchanged.
    _require_clipping_region_ref_extension(
        requested_payload.get("clipping_region_ref"),
        adjusted_payload.get("clipping_region_ref"),
    )
    adjustable = {
        "target_box",
        "hard_bounds",
        "metadata",
        "clipping_region_ref",
    }
    protected_requested = {
        key: value
        for key, value in requested_payload.items()
        if key not in adjustable
    }
    protected_adjusted = {
        key: value
        for key, value in adjusted_payload.items()
        if key not in adjustable
    }
    if protected_requested != protected_adjusted:
        changed = sorted(
            key
            for key in set(protected_requested) | set(protected_adjusted)
            if protected_requested.get(key) != protected_adjusted.get(key)
        )
        raise EffectiveLayerRendererError(
            "renderer changed protected effective plan fields: "
            + ",".join(changed)
        )


def _require_clipping_region_ref_extension(
    requested: Any,
    adjusted: Any,
) -> None:
    requested_ref = dict(requested) if isinstance(requested, Mapping) else {}
    adjusted_ref = dict(adjusted) if isinstance(adjusted, Mapping) else {}
    renderer_owned_keys = {"shape_aware_safe_box", "visual_slot_box"}

    if any(
        adjusted_ref.get(key) != value
        for key, value in requested_ref.items()
    ):
        raise EffectiveLayerRendererError(
            "renderer changed protected clipping-region references"
        )
    unexpected = sorted(set(adjusted_ref) - set(requested_ref) - renderer_owned_keys)
    if unexpected:
        raise EffectiveLayerRendererError(
            "renderer added unsupported clipping-region references: "
            + ",".join(unexpected)
        )
    for key in renderer_owned_keys & set(adjusted_ref):
        value = adjusted_ref.get(key)
        if (
            not isinstance(value, list)
            or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
            or value[2] <= 0
            or value[3] <= 0
        ):
            raise EffectiveLayerRendererError(
                f"renderer emitted an invalid {key} clipping reference"
            )
