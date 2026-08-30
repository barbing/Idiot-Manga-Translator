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
- AuthorizedSourceStyleView and axis-specific StyleEvidence expose only
  parent-authorized source pixels and measurements. ParentStyleArbitrator is the
  sole automated resolved-style owner.
- The page-ordered style-context cache transports qualified evidence from
  already committed pages. It does not resolve styles and never receives user
  overrides, rendered pixels, layout results, or cleanup judgments.
- RenderLayerAdapter, RenderLayoutPlanner, TypesettingEngine, the shaped-glyph
  rasterizer, parent effects, and RendererCompositor each own one downstream
  rendering decision boundary.
- Before parent finalization, OCR may remain review/conservation evidence without creating downstream work. Once a parent is finalized as executable, OCR, translation, cleanup, and rendering form a mandatory top-down chain and no later diagnostic can cancel that parent.
- The normal required-stage result is valid-but-imperfect: quality limitations are retained as editable artifacts plus diagnostics. Only technical inability to produce a valid required artifact may fail; that failure is terminal and no dependent stage or later page executes.
- The controller commits the completed page and its already prepared style-cache
  delta atomically before emitting `page_ready`.

The target default chain is:

```text
BubbleDetection typed evidence
  -> TextAreaPlan semantic units
  -> scoped CTD/TextForegroundSegmentation projection
  -> component authorization map
  -> scoped OCR source capture
  -> provisional TextBlockHierarchy parent obligations
  -> parent-boundary OCR source capture
  -> TextBlockHierarchy finalized execution units
  -> ParentExecutionBundle
  -> parent-keyed translation assignments
  -> SourceGlyphMask
  -> CleanupJob
  -> CleanupMask
  -> initial RenderEligibility
  -> CleanupPlan
  -> CleanupBackend
  -> CleanupResult
  -> CleanupProof
  -> cleanup commit and RenderEligibility diagnostic update
  -> immutable CleanedPageBase
  -> AuthorizedSourceStyleView and axis-specific StyleEvidence
  -> ParentStyleArbitrator and qualified style-cache delta
  -> RenderLayerAdapter and RenderLayerPlan
  -> RenderLayoutPlanner and TypesettingEngine
  -> shaped-glyph rasterization, optional effects, and RendererCompositor
  -> atomic page/project/style-cache checkpoint
  -> page_ready
```

## Stage Ownership

### UI and Controller

`app/main.py` is a thin production bootstrap. It calls
`app.ui.application_coordinator.create_gui_application_window`, which constructs
the single `app.ui.shell.main_window.YomiFrameMainWindow` and connects typed
project, settings, run, Preview, failure, recovery, cancellation, and close
signals exactly once. The legacy monolithic window and Page/Region Review owners
remain compatibility modules, not production authorities.

Cross-platform behavior is composed once by the production bootstrap through
`app.platform_services` rather than inferred from a generic `use_gpu` flag.
`PlatformServices` binds immutable
platform identity, standard paths, credential persistence/resolution, measured
compute capabilities, and the nine-item runtime asset catalog. Windows selects
Credential Manager and CUDA-capable policies; macOS selects Keychain, MPS,
CoreML, Metal/native llama.cpp, and unified-memory admission. CPU fallback is a
reported backend decision, not evidence that acceleration succeeded.

New-folder preparation has an application-owned pre-run projection: supported
source images become naturally ordered queued page rows with thumbnails,
selection, activity context, and a source canvas before pipeline artifacts
exist. These rows are workflow presentation only and do not fabricate detection,
OCR, parent, cleanup, style, eligibility, or render evidence.

Provider activation is likewise application-owned. The coordinator validates a
GGUF file, an Ollama endpoint/model, or an authenticated DeepSeek model catalog
through a settled worker and marks the exact public profile configuration ready.
The shell enables Start only for the selected ready profile. API credentials are
transient during testing and portable settings retain only an opaque credential
reference. Resolution is explicit through the native platform store (Windows
Credential Manager or macOS Keychain) or a named environment-variable
reference. Output Defaults supplies fallback presentation; it is not
a renderer command and cannot override effective per-parent style evidence or
user edits.

The desktop UI gathers user selections and calls typed application or edit
interfaces. The controller orchestrates functional module execution and
preserves the TextAreaPlan contract when regions are serialized, rehydrated,
filtered, or routed; the GUI never mutates controller or pipeline internals.

The controller should not turn route intent into translation authority. A region is translatable only when TextAreaPlan has explicitly marked OCR, translation, cleanup, and render eligibility as appropriate for a cleanup-translatable semantic state. Review-only OCR conservation does not create translation or cleanup authority.

In the current root-parent-child architecture, the controller promotes finalized
parent obligations into `ParentExecutionBundle` records before downstream
execution. Translation input rebuilding, cleanup job creation, render
eligibility, and renderer entry use that parent-bundle path for every executable
parent. Region-shaped records remain compatibility envelopes and audit records;
source child regions are evidence for a parent, not independent downstream
execution owners or a fallback execution path.

GUI user topology is an application projection over those immutable records.
`ParentSourceEvidenceMappingV1` binds exact parent/root/bundle identities,
automatic fingerprints, integer bboxes, OCR and target text/fingerprints,
reading order, roles, and page identity. Merge replaces compatible mapped source
slots; Split assigns every mapped source member to exactly one child and fails
if a boundary cuts evidence. Mechanical effective render-plan projection may
reuse only the bound automatic layers and substitute effective identity/text/
geometry. It cannot create a bundle, source-style fact, cleanup proof, or render-
eligibility decision.

Add Custom Scope is different: it first records only pending geometry/workflow
intent. An explicit OCR revision may establish source text for that exact scope,
but neither Add nor typed text makes it executable. Free-form source/target text
is accepted only as an override on retained mapped or selected-revision evidence.

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

The projection output feeds component authorization and cleanup masks. Projection quality is diagnostic after parent finalization; it cannot recolor semantic state or cancel the parent cleanup task.

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

Scoped OCR first consumes TextAreaPlan-eligible projected text areas. The
controller then builds a provisional hierarchy so the parent-boundary OCR owner
can recognize the exact parent obligation rather than promoting detector
fragments into downstream identity. A second hierarchy build incorporates that
parent-owned source evidence and publishes the finalized execution units used to
create `ParentExecutionBundle` records.

Some compatibility or review-conservation regions may be OCR-eligible while
remaining blocked from translation, cleanup, and rendering. Known
SFX/decorative/art regions do not enter the normal translation path unless a
future feature explicitly defines SFX translation support.

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
  -> initial RenderEligibility
  -> CleanupPlan
  -> CleanupBackend
  -> CleanupResult
  -> CleanupProof
  -> cleanup commit and RenderEligibility diagnostic update
  -> immutable CleanedPageBase
```

`CleanupMask` is a strict consumer. It should only erase components that upstream authorization and parent ownership made executable. Unknown, protected SFX/decorative, art, and non-text components must remain non-executable.

The initial render-eligibility contract is built after SourceGlyph, cleanup-job,
and cleanup-mask contracts and is supplied to cleanup planning. Cleanup runtime
and the upstream image commit may append warning/diagnostic state to that same
contract. After parent finalization those diagnostics cannot suppress a required
parent. Successful cleanup publication produces the immutable
`CleanedPageBase` consumed by style observation and rendering.

### Source Style Observation, Arbitration, and Cache

`AuthorizedSourceStyleView` binds one parent to read-only original pixels plus
accepted component-authorized foreground and exact provenance. Style observers
produce independent `StyleEvidence` for font identity/design, thickness, width,
source scale, fill, outline, orientation, rotation, and shadow, with explicit
support or abstention per axis. They do not decide the final style.

`ParentStyleArbitrator` is the sole automated resolved-style owner. It produces
one complete immutable style per parent, applies deterministic field-local
fallbacks where evidence is unavailable, and may consume only a validated
snapshot of qualified evidence from already committed pages. The page-ordered
style-context cache transports that evidence and an already prepared current-page
delta; it never resolves styles, observes future pages, or accepts rendered,
layout, fit, cleanup, fallback, or user-edit output.

User style and layout edits are projected mechanically after automated style
resolution. They never become `StyleEvidence`, change automated arbitration, or
become style-cache donors.

### Rendering

Rendering composes translated text after cleanup and style resolution. The sole
production facade is `render_parent_execution_bundles()`. It stamps parent audit
identity and calls `RenderLayerAdapter` to perform a lossless one-parent-to-one-
`RenderLayerPlan` conversion. `PageRenderExecutor` then sequences
`RenderLayoutPlanner`, `TypesettingEngine`, `InkBoundLayoutFitter`, shaped-glyph
rasterization, optional parent effects, and the draw/commit-only
`RendererCompositor` against one immutable `CleanedPageBase`.

This path must preserve the full translated text or fail the page transaction
for a genuine technical construction error. It must not silently drop
characters, reinterpret semantic scope, perform cleanup, re-resolve style, or
commit a partial required parent. Fit/readability limitations remain diagnostics
and user-editable quality issues rather than render-admission authority.

English uses `target-presentation:en:v2`. For an exact authorized speech
container, its automatic domain is the full container shape while the source
side/center remains an alignment prior. `RenderLayoutPlanner` converts the
exact polygon into an ephemeral pre-effect actual/comfort row-capacity profile;
`TypesettingEngine` performs whole-word variable-width line selection and
per-line alignment; the existing compositor still applies the one final parent
rotation and draw/commit transaction. Chinese retains the existing CJK path.
Missing or conservative polygon authority falls back to the prior rectangular
layout without pixel inference becoming semantic authority.

Horizontal Latin sans roles resolve through the managed Noto Sans 2.015
variable face at the locked condensed width axis; logical source/user role and
weight tiers remain unchanged, and a missing Latin asset falls back to the
bundled Noto CJK role rather than an operating-system font. Shape-band
placement normalizes both polygon and source alignment into the pre-effect
frame and compares a bounded near-size window so a trivial size gain cannot
cause material center drift. A vertical-source speech parent containing one
long Latin lexical word may use one atomic 90-degree local display orientation
when no well-centered horizontal candidate exists; the word is never split or
hyphenated, and the normal parent residual rotation still executes exactly
once in the compositor.

For an unlocked English/Latin shape-band parent, the typesetter may also
compare the rounded upper endpoint of the style-owned preferred interval with
the central/downward result. It retains that endpoint only when text completeness, lexical/punctuation
quality, actual/comfort containment, and centering are non-regressing. When a
selected leading-comma phrase reflow alone consumes the eroded comfort inset,
the endpoint may remain eligible only if predicted raster ink retains a
one-pixel guard inside every actual shape band; that exception is explicit in
the candidate audit and does not relax hard contour containment. The
interval cannot be expanded or searched at intermediate sizes by the renderer,
and user-locked, CJK, legacy, and
profile-unavailable paths remain unchanged.

Automatic horizontal English break records may carry deterministic soft phrase
evidence for a leading comma boundary and unambiguous closed-class attachment.
The existing `LineBreakPlanner` remains the sole selector; hard fit and exact
text remain higher authority. English preferred-break and keep ranks are used
only inside an already fixed line-count path and are excluded from canonical
cross-topology/cross-size quality. The evidence is same-size, locale-scoped,
and cache-bound, and cannot change legality, add a layout attempt, or affect
CJK, manual edits, geometry, style, effects, or composition.

The renderer consumes cleanup results and render-eligibility decisions. It must not generate cleanup masks, choose cleanup classes, select cleanup backends, or mutate source cleanup locally in normal operation.

The renderer does not validate cleanup completeness. It consumes the supplied
`CleanedPageBase` and every render-required parent. Cleanup/proof and
render-eligibility records remain diagnostics; they cannot suppress parent
execution. A required layer failure fails the page transaction instead of
committing a successful partial page.

### Manual Cleanup Revisions

Manual cleanup is a separate application-owned edit workflow, not an alternate
automatic cleanup authority. `ManualCleanupService` binds an explicit typed mask
request to one selected immutable `CleanedPageBase`, invokes the existing cleanup
backend for Preview, and commits only after user confirmation. A successful
commit publishes a new immutable `CleanedPageBase` revision and a typed
`ManualCleanupReceipt`.

The service never inpaints the rendered final page, claims automated
`CleanupProof`, changes semantic authorization, or silently reapplies a preview
to a different/stale base hash. Cancelled or stale previews do not become current
project state.

### Project Checkpoint and Page Publication

After rendering, the controller prepares the completed page record and the
already produced current-page style-cache delta. `app.io.project_checkpoint`
owns one atomic durable commit of those opaque values, including storage framing,
hashes, rollback, and committed-prefix recovery. It does not interpret or rerun
pipeline, style, renderer, or GUI policy. `app.io.project` owns compatible load
and the final full-project materialization used by review and diagnostics.

The page record and style-cache delta advance together. A failed commit leaves
the previous committed prefix loadable and terminates the run before later pages.
The controller emits `page_ready` only after the checkpoint receipt succeeds.

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

There is no raw-region rendering fallback. If canonical graph output contains no
render layers, the controller copies the valid `CleanedPageBase` through the
explicit no-layer path. If an authorized workflow region exists without the
required executable parent bundle, the controller raises a hierarchy/identity
contract failure rather than rendering the region independently.

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

Python 3.10 is the supported source interpreter on Windows and macOS. Windows
can install `requirements.txt` in an isolated Conda environment; its platform
marker retains `onnxruntime-gpu`. macOS uses `environments/macos.yml`, which
installs the matching PyICU/ICU pair plus pinned Conda-native `llama.cpp` and
`llama-cpp-python` packages before installing the remaining Python
requirements. The Darwin requirements branch uses `onnxruntime` and excludes
the pip llama binding so the default setup does not require an undeclared Xcode
source build. Available providers are still inspected at runtime.

Backend selection is capability-based and component-specific:

- Torch prefers CUDA, then MPS, then CPU.
- ONNX Runtime prefers CUDA, then CoreML, then CPU.
- llama.cpp executables resolve from an explicit override, the active
  environment/PATH, native platform names, and retained Windows `.exe` layouts.
- Paddle's `llama-server` device listing and GGUF translation's
  `llama-cpp-python` GPU-offload capability are measured independently. An
  unknown or CPU-only extension forces GGUF layer 0 and host-RAM admission even
  when NVIDIA memory is present.
- PaddleOCR-VL uses CUDA llama.cpp assets on Windows and the Conda/native Metal
  runtime on macOS.
- MPS/CoreML/Metal admission uses a shared unified-memory budget constrained by
  both available system RAM and PyTorch's recommended Metal working set.
- CUDA retains separate VRAM and host-RAM budgets.

Runtime metadata records requested and selected backends/providers and typed
fallback reasons. A CPU-only or remote DeepSeek plan is not blocked merely
because an accelerator probe is absent.

BubbleDetection compares active providers with the requested CoreML/CUDA/CPU
chain rather than treating every non-CUDA provider as fallback; session
construction retries CPU once with a typed initialization reason. Torch cache
release likewise follows CUDA or MPS through the shared compute policy. The NER
loader requires a complete local model/tokenizer snapshot and uses
`local_files_only=True`, avoiding a background Hub conversion thread during
normal offline startup and shutdown.

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

After first paint the Settings runtime catalog presents nine fixed families
without automatically importing model frameworks or starting a model server:
ComicTextDetector,
bubble evidence, PaddleOCR-VL, MangaOCR, cleanup inpainting, NER, YuzuMarker,
the Noto CJK pack, and PyICU. The same catalog owns row copy, checker/preparer
method names, platform remediation, and managed-download availability. On Mac,
Paddle downloads only its model/projector and requires the native Conda
executable; Windows retains the two CUDA archives. User-selected LLM
translation models are not fixed catalog assets. **Verify all** runs the
model-free probe explicitly. Start blocks only for a selected asset that a
current receipt proves missing; MangaOCR, NER, or another unselected catalog
row cannot veto the run. With no current receipt, the normal owning stage keeps
its fail-closed model/runtime startup contract.

Application preferences migrate only exact untouched historical defaults on
macOS: the legacy Windows project-root default becomes the platform Documents
project root, and default Undo/Redo/Preview bindings become Command-based.
Custom paths and shortcut values are preserved. Qt supplies the UI font;
renderer defaults and heuristic fallbacks use the installed Noto CJK pack
rather than Microsoft YaHei. Frameless macOS edges call Qt
`startSystemResize`, while the guarded Windows `WM_NCHITTEST` path remains the
Windows fallback.

The checked-in PyInstaller/bundled-ICU packaging path is Windows-only. macOS is
currently supported as a source/Conda execution path, not as a signed or
notarized application bundle.

Historical page-specific model-fusion/debug assists are removed from the
production pipeline. No environment flag may revive them as semantic,
hierarchy, translation, cleanup, eligibility, style, or renderer authority.

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
- Do not create user topology or render authority from typed text. Require an
  exact OCR/detection mapping or selected owner revision and preserve it through
  projection, History, and reload.

## Recommended Lightweight Checks

Syntax checks for edited Python modules:

```powershell
python -m py_compile app/pipeline/bubble_detection.py app/pipeline/text_area_plan.py app/pipeline/text_block_hierarchy.py app/pipeline/parent_execution_bundle.py app/pipeline/controller.py
```

Run focused tests for the owner changed by the work and report the exact command
and result. Do not publish or recommend a command for a test module that the
source tree does not contain.

Full validation requires the task-specific contract, runtime, and visual
workflow described above. Bare repository-wide `pytest`, syntax checks alone,
or test counts alone do not establish release readiness.
