# Manga Translator Pro - Technical Notes (v1.0.0)

## Overview
Manga Translator Pro is a Windows desktop app for batch manga translation. It detects text regions,
OCRs the Japanese, translates to the target language, removes the original text, and renders
translated text back into the page. The app stores results in a project JSON file and outputs
translated images to the user-selected export folder.
When Auto-Glossary is enabled, it also builds a `style_guide.json` for consistent noun translation.

## Architecture
- UI: PySide6 (Qt) dashboard with Home/Queue/Library/Settings pages.
- Pipeline: `PipelineController` orchestrates detect -> OCR -> translate -> render.
- Models:
  - Text detection: ComicTextDetector (optional)
  - OCR: MangaOCR / PaddleOCR
  - Translation: Ollama or GGUF via llama-cpp-python
  - Inpainting: fast (OpenCV) or AI inpainting (LaMa - Large Mask Inpainting)
  - Auto-Glossary: MeCab (fugashi) with optional LLM Hybrid Discovery
  - Deep Scan: separate backend/model for extraction-focused analysis

Key modules:
- `app/pipeline/controller.py` - pipeline orchestration, batching, caching
- `app/nlp/mecab_extractor.py` - MeCab proper noun extraction and alias grouping
- `app/render/renderer.py` - text removal and text rendering
- `app/detect/*` - text detection backends
- `app/ocr/*` - OCR backends
- `app/translate/*` - translators (Ollama, GGUF)
- `app/ui/*` - UI layout, theme, and dialogs

## Code Structure (with explanations)
- `app/main.py`
  - App entry point. Creates the Qt application, applies theme, and launches `MainWindow`.
- `app/ui/main_window.py`
  - Main UI layout and navigation (Home/Queue/Library/Settings).
  - Wires UI events to pipeline actions.
  - Manages thumbnails, live inspection panel, and queue table updates.
- `app/ui/theme.py`
  - Dark/light palette and stylesheet definitions used by the UI.
- `app/ui/page_review.py` and `app/ui/region_review.py`
  - Review dialogs for side-by-side page inspection and per-region edits.
- `app/pipeline/controller.py`
  - Orchestrates the translation pipeline:
    - Loads pages, dispatches detection, OCR, translation, rendering.
    - Tracks per-page timing, progress, and errors.
    - Emits UI-friendly status updates.
- `app/detect/*`
  - Text detector backends (e.g., comic text detector).
  - Each detector returns regions with bounding boxes/polygons.
- `app/ocr/*`
  - OCR backends (MangaOCR / PaddleOCR).
  - Returns OCR text per region with confidence.
- `app/translate/*`
  - Translators (Ollama or GGUF/llama-cpp).
  - Handles batching and prompt formatting.
- `app/render/renderer.py`
  - Renders translated text into the original image.
  - Handles bubble/background cleaning and text placement.
  - Applies color selection and stroke for contrast.
- `app/io/project.py`
  - Reads/writes project JSON files.
  - Used by the pipeline and review dialogs.

## Pipeline Flow (per page)
1. Load image (optionally downscale for detection only).
2. Detect text regions (speech bubbles + background text).
3. OCR each region (crop-based).
4. Translate unique OCR strings (batched).
5. Render:
   - Remove original text (bubble fill or inpaint).
   - For background text, fill a local color patch to avoid mosaic artifacts.
   - Place translated text in region bounds.
6. Save output image and update project JSON.

### Pipeline - Detailed Walkthrough
This section describes how the pipeline moves data through the system and where each step
is implemented in code.

1) **Image load & normalization**
   - The controller reads each image path and loads it for processing.
   - Detection can run on a resized version for speed, but rendering uses the full-res image.
   - Goal: speed without losing output quality.

2) **Text detection**
   - A detector backend (e.g., ComicTextDetector) returns region proposals.
   - Each region contains a bbox and polygon, used later for OCR and rendering.
   - The controller stores these regions in the per-page project structure.

3) **OCR**
   - Each region is cropped and passed to the selected OCR engine.
   - The OCR output is attached to the region as `ocr_text` with `confidence.ocr`.
   - If OCR fails, the region is flagged and the UI log shows the error.

4) **Translation**
   - The controller de-duplicates OCR strings and batches them for translation.
   - The translator backend returns translated strings, which are re-applied to regions.
   - Translation is cached to avoid repeated calls for identical text.
   - Auto-Glossary runs in parallel (MeCab-only by default), and can optionally use
     Hybrid Discovery (LLM) for deeper extraction.

5) **Rendering**
   - Rendering uses the region bbox/polygon and translation text.
   - Speech bubbles:
     - Prefer a clean fill for solid bubbles, then draw translated text.
   - Background text:
     - Fill with sampled local color to avoid mosaic artifacts.
     - Force text color (white on dark backgrounds, black on light backgrounds).
   - Text layout uses region size to estimate font size and line breaks.

6) **Output**
   - The translated image is saved to the export folder.
   - The updated project JSON is saved with render settings and confidence scores.
   - Auto-Glossary saves `style_guide.json` to the export folder.
   - A consistency check can flag early pages that predate the final glossary.

## Project JSON
Each page contains a list of regions:
```json
{
  "region_id": "r000",
  "bbox": [x, y, w, h],
  "polygon": [[[x, y], ...]],
  "type": "speech_bubble|background_text",
  "ocr_text": "...",
  "translation": "...",
  "confidence": {"det": 1.0, "ocr": 1.0, "trans": 1.0},
  "render": {"font": "Microsoft YaHei", "font_size": 0, "line_height": 1.2, "align": "center"},
  "flags": {"ignore": false, "bg_text": false, "needs_review": false}
}
```

### JSON Block Explanation
- `region_id`: Stable ID for tracking a text region across UI review and re-rendering.
- `bbox`: Axis-aligned bounding box `[x, y, w, h]` used for layout.
- `polygon`: Exact region outline; used for precise fill and text placement.
- `type`: Region type (`speech_bubble` or `background_text`) which changes render logic.
- `ocr_text`: Raw OCR output from the selected engine.
- `translation`: Final translated text used by the renderer.
- `confidence`: Detection/OCR/translation confidence fields (float 0-1).
- `render`: Rendering preferences for this region (font, line height, color, stroke).
- `flags`: Workflow flags (ignored regions, background text, or needs review).

## Auto-Glossary
When enabled:
- OCR text is accumulated for noun discovery.
- MeCab extracts proper nouns and groups aliases.
- Translated names are stored in `style_guide.json`.
- Optional Hybrid Discovery (LLM) can enrich the glossary.
- A consistency check can suggest re-translation of early pages.

## Rendering Strategy
Rendering is handled by `app/render/renderer.py`:
- Speech bubbles:
  - **Adaptive Dilation**: Uses statistics (stddev) to apply aggressive dilation on uniform bubbles and minimal dilation on complex backgrounds.
  - White Fill: For pure white bubbles, simple fill.
  - Inpainting:
    - **LaMa (AI)**: Used for complex backgrounds or when `inpaint_mode="ai"`. Handles large text removal better than CV2.
    - **CV2 Telea/NS**: Fallback for simple cases or when AI is disabled.
- Background text:
  - **Vertical Text**: Detects tall regions (aspect ratio &lt; 0.8) and renders text vertically.
  - Local color sampling fill to avoid mosaic artifacts.
- Text layout:
  - Auto font sizing to fit region.
  - CJK-aware punctuation normalization.
  - Line wrapping and centering.

### Rendering - Code Blocks Explained
Key blocks in `renderer.py` and their purpose:
- **Region classification**
  - Determines if a region is a speech bubble or background text, based on detector hints
    and luminance statistics. This controls the fill strategy.
- **Background fill**
  - Uses local color statistics to fill text areas instead of inpainting to avoid blur.
- **Bubble fill**
  - For solid bubbles, uses a clean fill; for complex bubbles, falls back to inpaint.
- **Font sizing**
  - Computes a font size that fits the region and avoids overflow.
- **Text placement**
  - Centers or aligns text based on the region’s bbox and chosen alignment.

## UI Pages
- Home: progress, thumbnails, log, region review.
- Queue: list of pages and status.
- Library: completed pages list.
- Settings: model selection, rendering, performance options.

### UI Code Blocks Explained
- **Navigation**
  - The left sidebar toggles pages in a stacked widget to switch between Home/Queue/Library/Settings.
- **Home grid**
  - Thumbnails are loaded lazily, with status overlays (processing/done/error).
- **Live Inspection**
  - Clicking a thumbnail updates the inspector panel with OCR + translation for quick spot-checking.
- **Review Dialogs**
  - **Page Review**: 
    - Split view (Original vs Translated).
    - **Resizable Images**: Images scale dynamically with window size.
    - **Adjustable Layout**: Main splitter ratio favors images (5:1) for better visibility.
    - Editable translation table with "Needs Review" flagging.

## Performance
Target: 6 pages under 2 minutes on the reference PC.
Key optimizations:
- Detection downscaling for speed, render on full-res image.
- Translation cache for repeated lines.
- Batched translation requests.

## Error Handling
The UI shows explicit error messages and does not silently ignore failures.
Common issues:
- Missing torch (MangaOCR fallback to PaddleOCR).
- Missing models (prompt user to install).

### Error Handling - Code Blocks Explained
- **Dependency checks**
  - Model init is guarded; dependency failures are surfaced as readable UI messages.
- **Per-page failures**
  - Errors are recorded in the queue table and log, not swallowed.
- **Graceful fallbacks**
  - OCR and translator backends can fall back when possible.

## Local Models
Models are not committed to the repository. Place local models in:
```
models/
  comic-text-detector/
  sakura/
  ...
```

## Running
From repo root:
```
python -m app.main
```

## Testing
- Small changes: `python -m py_compile <files>`
- Full flow: run the app, load `Test Manga`, verify:
  - No startup errors
  - Output images show translated text in bubbles
  - Project JSON saved
