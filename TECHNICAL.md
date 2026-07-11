# YomiFrame Technical Notes

This document describes the technical contracts behind the current specialized modular pipeline. The README is the broad project overview; this file owns implementation-level responsibilities, authorization fields, cache contracts, validation mechanics, and stage-boundary rules.

## Architecture Principles

YomiFrame separates semantic authority from pixel evidence and from downstream execution:

- BubbleDetection/TextAreaPlan own semantic authority.
- BubbleDetection normalizes model evidence from the bubble/text-area ensemble.
- TextAreaPlan adjudicates that evidence into typed semantic text units and standardized downstream eligibility fields.
- ComicTextDetector/TextForegroundSegmentation supplies text pixels inside scoped regions.
- TextBlockHierarchy normalizes physical roots, parent text obligations, and child source evidence into a finalized graph view.
- ParentExecutionBundle converts finalized parent obligations into the downstream execution contract.
- CleanupMask consumes upstream authorization and foreground projection; it does not infer speech, background, SFX, art, or review semantics from local component geometry.
- OCR, translation, cleanup, and rendering require explicit TextAreaPlan fields instead of accepting route intent alone. OCR may be allowed for review/conservation while translation, cleanup, and rendering remain blocked.

The target default chain is:

```text
BubbleDetection typed evidence
  -> TextAreaPlan semantic units
  -> scoped CTD/TextForegroundSegmentation projection
  -> component authorization map
  -> OCR source capture
  -> TextBlockHierarchy finalized execution units
  -> ParentExecutionBundle
  -> parent-keyed translation assignments
  -> SourceGlyphMask
  -> CleanupJob
  -> CleanupMask
  -> CleanupPlan
  -> CleanupBackend
  -> CleanupResult
  -> CleanupProof
  -> RenderEligibility
  -> parent-bundle renderer final composition
```

## Stage Ownership

### UI and Controller

The desktop UI gathers user selections and calls the pipeline controller. The controller orchestrates module execution and preserves the TextAreaPlan contract when regions are serialized, rehydrated, filtered, or routed.

The controller should not turn route intent into translation authority. A region is translatable only when TextAreaPlan has explicitly marked OCR, translation, cleanup, and render eligibility as appropriate for a cleanup-translatable semantic state. Review-only OCR conservation does not create translation or cleanup authority.

In the current root-parent-child architecture, the controller must promote finalized parent obligations into `ParentExecutionBundle` records before downstream execution. Translation input rebuilding, cleanup job creation, render eligibility, and renderer entry all use the parent bundle path when bundles are present. Legacy region records remain compatibility and audit records; source child regions are evidence for a parent, not independent downstream execution owners.

### BubbleDetection

`app/pipeline/bubble_detection.py` is the upstream model-evidence provider. It combines speech-bubble and text-area evidence from the available bubble detection models, including Kitsumed-style speech-bubble output and Ogkalu labels such as `bubble`, `text_bubble`, and `text_free`.

BubbleDetection is responsible for:

- preserving raw model labels
- normalizing candidate kind
- stamping evidence strength
- recording source evidence IDs
- auditing edge and clipping context
- computing neighboring speech context
- exposing first-class reason codes for speech and free-text evidence
- including semantic contract identity in runtime and cache metadata

BubbleDetection does not by itself authorize cleanup. It supplies typed evidence to TextAreaPlan.

### TextAreaPlan

`app/pipeline/text_area_plan.py` is the semantic authority layer. It converts evidence into typed text units before CTD/component projection.

TextAreaPlan is responsible for:

- speech-bubble text authorization
- background/title/narration text authorization
- caption authorization
- SFX/decorative protection
- art/non-text protection
- review/unknown quarantine
- deterministic eligibility fields for OCR, translation, cleanup, and rendering
- explicit authorization state, basis, and origin

TextAreaPlan must distinguish a candidate or review state from executable semantic authority. A high-confidence single-model result may be promoted only through documented constraints, such as Ogkalu speech evidence with page-edge/clipping support, neighboring speech context, and no protected conflict.

### Text Pixel Projection

ComicTextDetector/TextForegroundSegmentation provides text-pixel evidence within TextAreaPlan scopes. It may refine component boundaries and foreground pixels, but it must not become the semantic owner of speech, background, SFX, or review classifications.

The projection output feeds component authorization and cleanup masks. Projection quality may affect executability, but projection readiness should not recolor semantic state.

### Root / Parent / Child Hierarchy

`app/pipeline/text_block_hierarchy.py` owns the explicit text-block graph after BubbleDetection/TextAreaPlan, CTD projection, and OCR source capture have produced evidence.

The hierarchy separates:

- root blocks: physical text containers such as speech bubbles, caption/background boxes, unknown fallback areas, or protected SFX/decorative containers
- parent logical text units: executable text obligations that should be translated, cleaned, and rendered as coherent units
- child recognized text segments: detector/OCR fragments represented by a parent or excluded as non-workflow/protected evidence

`TextBlockHierarchyResult.finalized_execution_units()` is the canonical graph view for downstream handoff. It exposes active translation parents, punctuation parent obligations, blocked/unresolved parents, represented children, and excluded non-workflow children. It is not a renderer or cleanup tool; it defines ownership and conservation of text obligations.

### ParentExecutionBundle

`app/pipeline/parent_execution_bundle.py` converts finalized hierarchy parents into `ParentExecutionBundle` records.

A parent execution bundle carries:

- `bundle_id`, `parent_id`, `graph_parent_id`, and `root_id`
- parent `state`, `role`, source text, OCR provenance, and source-quality action
- `execution_region`, `parent_bbox`, cleanup target bbox, render allowed area, and root bbox
- represented child ids and source region ids
- translation, cleanup, render, SourceGlyph, cleanup-mask, render-decision, and renderer-audit ids
- render style contract fields such as orientation, wrap mode, stroke, fill color, size hints, and style class

The bundle's `execution_region` is a parent-owned compatibility record for downstream code that still accepts region-shaped dictionaries. It explicitly marks `execution_region_authority = parent_execution_bundle`, `parent_execution_authoritative = True`, and `source_region_evidence_only = True`.

Downstream modules must not create new execution units from child/source regions after this handoff. If a source region has to be inspected, it is evidence attached to the parent bundle.

### OCR

OCR consumes TextAreaPlan-eligible parent regions and projected text areas. Some compatibility or review-conservation regions may be OCR-eligible while still blocked from translation, cleanup, and rendering. Known SFX/decorative/art regions should not enter the normal translation path unless a future feature explicitly defines SFX translation support.

OCR errors should be diagnosed separately from semantic authorization errors. If visible text was never authorized upstream, OCR cannot fix the missed region.

### Translation and NLP

Translation consumes source text from eligible parent execution bundles. Assignment identity is parent-keyed; source text may be used as a cache key, but it must not replace parent identity. The NLP layer handles glossary, name memory, style guidance, and consistency. It should not promote review/unknown regions into translation merely because text was detected.

### Cleanup

Cleanup begins only after upstream semantic authorization, parent execution bundling, and projection have produced approved source-glyph and foreground evidence.

The cleanup chain is:

```text
ParentExecutionBundle
  -> SourceGlyphMask
  -> CleanupJob
  -> CleanupMask
  -> CleanupPlan
  -> CleanupBackend
  -> CleanupResult
  -> CleanupProof
```

`CleanupMask` is a strict consumer. It should only erase components that upstream authorization and parent ownership made executable. Unknown, protected SFX/decorative, art, and non-text components must remain non-executable.

### Rendering

Rendering composes translated text after cleanup. The primary entry point for current production output is `render_parent_execution_bundles()`, which converts bundles to parent-owned execution regions and stamps renderer audit identity. It must preserve the full translated text or produce explicit evidence when text cannot fit. It should not silently drop characters, overflow unreadably, or reinterpret semantic scope.

The renderer consumes cleanup results and render-eligibility decisions. It must not generate cleanup masks, choose cleanup classes, select cleanup backends, or mutate source cleanup locally in normal operation.

## Semantic Authorization Contract

Downstream modules receive standardized fields rather than reading historical confidence-tier names or marker strings. The contract includes:

- CTD scope
- OCR eligibility
- translation eligibility
- render eligibility
- cleanup executability
- finalized root id
- finalized parent id
- parent execution bundle id
- parent-owned execution region
- represented child/source evidence ids
- explicit authorization state
- authorization basis
- origin/provider metadata

Important semantic states include:

- `cleanup_translate_speech`
- `cleanup_translate_background`
- `cleanup_translate_caption`
- `protect_sfx_decorative`
- `protect_art_or_non_text`
- `review_unknown_not_cleanup`
- `outside_cleanup_scope`
- `ambiguous_component_owner`

The required invariant is that executable downstream work must come from explicit TextAreaPlan authority plus finalized parent ownership. Candidate-only signals, stale artifacts, SourceGlyph/projection artifacts, page geometry, bbox overlap, cleanup jobs, child fragments, and legacy regions do not create semantic authority or parent execution identity.

## Parent Execution Contract

The parent execution contract is the current boundary between graph construction and downstream execution.

### Contract Owner

`TextBlockHierarchyResult.finalized_execution_units()` owns the graph view. `ParentExecutionBundle` owns the downstream handoff shape. The controller is responsible for building bundles immediately after hierarchy generation and before translation, SourceGlyph generation, cleanup job construction, render eligibility, and rendering.

### Contract Consumers

Current parent-bundle consumers include:

- translation input rebuilding and `TranslationAssignment` creation in `controller.py`
- `generate_source_glyph_masks_for_parent_bundles()`
- `build_cleanup_job_candidates_for_parent_bundles()`
- `build_render_eligibility_decisions_for_parent_bundles()`
- `render_parent_execution_bundles()`
- review/rerender UI paths that rebuild bundles from persisted audit records

### Compatibility Boundary

Some internal APIs still accept region-shaped dictionaries. The parent execution layer handles this by producing `execution_region` records from bundles. These records are compatibility envelopes with parent identity, not a return to region-owned execution.

If no parent bundles are available, the controller still has a legacy region rendering path for compatibility/failure containment. That path should not be treated as the target architecture for new work.

### Identity Rules

- Parent id is the assignment identity for translation, cleanup, and render auditing.
- Root id identifies the physical container, not the source text obligation.
- Child ids identify source evidence represented by a parent.
- Source text can be used as a translation cache key only after parent identity is fixed.
- Punctuation-only parent obligations remain executable parent records when the graph classifies them as such.

## BubbleDetection Cache Contract

BubbleDetection cache entries must include the semantic evidence contract identity. Cache validation should reject payloads produced by an older semantic contract, even when model file paths and generic detector settings match.

This prevents stale evidence from a previous contract from being reused after changes to:

- semantic evidence schema
- neighboring speech context computation
- normalized Ogkalu/Kitsumed evidence interpretation
- authority-related provider behavior

The current cache contract is versioned in `app/pipeline/bubble_detection.py`. Cache invalidation is part of the semantic contract, not only detector performance.

## Validation Policy

Validation level depends on the changed module.

For BubbleDetection/TextAreaPlan/CTD/CleanupMask authorization changes:

- use a fresh mask-only run when explicitly approved
- inspect raw visual artifacts page by page
- compare original page, refined segmentation, semantic-unit/component authorization overlays, projection-quality overlay, protected/unknown overlay, clean foreground union, and clean erase union
- do not use contact-sheet prose, summary JSON, counters, or reports as proof

For cleanup runtime or inpainting changes:

- validate CleanupPlan, CleanupBackend, CleanupResult, and CleanupProof
- inspect inpainted output against original and masks
- confirm protected material is preserved and approved source text is erased

For OCR, translation, or rendering changes:

- run a real translation validation cycle when feasible
- inspect project JSON, parent execution bundles, OCR text, translated text, rendered output, overlays, and full pages
- check that rendered text contains the full `translated_text`

Mask-only validation cannot by itself prove end-to-end cleanup runtime, inpainting quality, rendering quality, or full translation readiness.

## Cleanup/Inpainting Readiness Boundary

Cleanup and inpainting should run only after upstream semantic authorization and text-pixel projection are accepted. Accepted cleanup masks are planned and executed through the configured cleanup backend, and proof records source-text removal, mask containment, broad-fill risk, and collateral-change evidence.

The renderer consumes the cleaned pre-render image, parent execution bundles, cleanup/proof metadata, and render-eligibility decisions. It should not generate cleanup masks, choose cleanup classes, select cleanup backends, or perform renderer-local cleanup mutation.

End-to-end quality is evaluated from rendered output pages, project and audit metadata, cleanup/proof consumption, rendered-text completeness, layout quality, SFX/decorative preservation, and performance.

## Auto-Glossary and Name Memory

The glossary system supports chapter-level translation consistency for names, titles, organizations, places, nicknames, and forms of address. It consumes OCR and translation context after region authorization has already happened.

Glossary enforcement must not mask upstream OCR or detection failures. If a name is missing because the text region was never authorized or OCR failed, the issue belongs upstream of glossary enforcement.

## Models and Environment

The project is Windows-first. For public setup, use Python 3.10 and install `requirements.txt` in an isolated virtual environment. YuzuMarker font detection uses `onnxruntime-gpu` by default, requests CUDA before CPU, and records the active provider plus any non-blocking CPU fallback reason. Conda is also acceptable when users need GPU-specific Torch, PaddlePaddle, or llama-cpp-python builds.

Local assets and caches are preferred over environment changes. Heavy new dependencies or mandatory extra models should not be added unless the roadmap or user explicitly authorizes them.

Main model families used by the pipeline include:

- Kitsumed and Ogkalu bubble/text-area detection models for BubbleDetection evidence
- ComicTextDetector/TextForegroundSegmentation for text-pixel projection
- PaddleOCR-VL GGUF as the default OCR engine and MangaOCR as an explicit selectable OCR engine
- local LLMs through GGUF and Ollama-compatible translation paths
- the fixed iopaint Anime Manga Big LaMA cleanup inpainting backend model
- NLP resources for glossary and name memory, including downloadable BERT NER assets when that optional path is enabled

Cleanup production code must not select among arbitrary `models/inpaint`
contents. The configured cleanup model id is provenance; the actual cleanup
backend resolves to the fixed iopaint model unless a future roadmap explicitly
changes the policy.

Startup pre-download covers fixed runtime assets such as text detection, bubble evidence, PaddleOCR-VL, MangaOCR, cleanup inpainting, and NER resources. User-selected LLM translation models are not part of the startup-required fixed asset set; they are selected or downloaded through the translation model UI.

Historical page-specific model-fusion/debug assists are not part of the
default pipeline. They must remain disabled unless
`MT_LEGACY_PAGE_SPECIFIC_ASSIST` and the specific diagnostic flag are both set.

## Performance Expectations

The default workflow must remain practical for local use. Translation-quality improvements should not make average processing time exceed roughly 30 seconds per page unless the user explicitly approves a slower path.

Performance-sensitive work should report:

- total runtime
- page count
- average time per page
- bottleneck stage when identifiable

## Development Guidelines

- Identify the owning stage before editing code.
- Do not fix semantic authorization defects inside CleanupMask.
- Do not fix OCR failures by changing translation prompts.
- Do not fix rendering overflow by changing OCR or semantic routing.
- Do not route SFX/decorative/art areas through normal OCR, translation, cleanup, or render paths unless explicitly required.
- Keep compatibility paths visible and temporary; do not let them become hidden alternate authority.
- Prefer deterministic contracts and explicit fallback states over ad hoc heuristics.

## Recommended Lightweight Checks

Syntax checks for edited Python modules:

```powershell
python -m py_compile app/pipeline/bubble_detection.py app/pipeline/text_area_plan.py app/pipeline/text_block_hierarchy.py app/pipeline/parent_execution_bundle.py app/pipeline/controller.py
```

Contract-focused unit tests, when relevant:

```powershell
python -m unittest app.tests.test_semantic_authority_contract
```

Full validation requires the task-specific visual or translation workflow described above.
