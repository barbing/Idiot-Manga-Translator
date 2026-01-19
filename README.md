# Manga Translator (Windows, Python)

Desktop manga translation tool with GUI. Import a folder of manga images, translate Japanese text into Simplified Chinese or English, render translations inside bubbles, and export images plus a project JSON you can edit and re-apply.

## Key Features
- Windows desktop GUI (PySide6)
- Batch import/export folders
- OCR engines: MangaOCR + PaddleOCR
- Text detection: ComicTextDetector
- Translators: Ollama (local models) and GGUF (llama.cpp)
- Style guide JSON (glossary + prompts)
- JSON project output for human edits + re-apply
- Progress/ETA + per-page timing
- Dark/Light theme

## Requirements
- Windows 10/11
- Python 3.10
- GPU recommended (RTX 4080 Super supported)

## Install Dependencies (pip)
```powershell
pip install -r requirements.txt
```

Notes:
- For GPU acceleration, prefer installing PyTorch/Paddle GPU builds via conda.
- AI inpainting (diffusers) is optional; see `requirements.txt` comments.

## Run the App
```powershell
python -m app.main
```

## Basic Workflow
1. Select **Manga Folder** (input images).
2. Select **Export Folder**.
3. (Optional) Set **Project JSON** output path.
4. Choose **Translator** (Ollama or GGUF).
5. Choose **OCR Engine** (MangaOCR recommended).
6. Press **Start**.

## Output
- Rendered images in the export folder.
- `project.json` with regions, OCR text, translations, render params, and flags.

## Re-apply Edited JSON
1. Manually edit `project.json`.
2. Click **Import JSON (Re-apply)** to render your edits without re-translation.

## Models
### Ollama
- Ensure `ollama` is installed and running.
- Use **Ollama Model** dropdown (auto-detect).

### GGUF
- Place `.gguf` models under `models/`.
- The app auto-scans `models/**.gguf` and lists them in **GGUF Model**.
- Example: `models/sakura/sakura-14b-qwen3-v1.5-q6k.gguf`

### ComicTextDetector Model (Local)
- Models are not tracked in the repository to keep size small.
- Place the ComicTextDetector model files under `models/comic-text-detector/`.
  (See the ComicTextDetector release page linked in the app error message.)

## Test Manga
You can use your own sample folder (e.g., `Test Manga/`) for validation.  
Note: local test data is not tracked in the repo.

## Troubleshooting
### App finishes instantly / no output
On Windows, OpenCV cannot read non-ASCII file paths with `cv2.imread`.
Use ASCII-only paths or the updated detectors (this repo already includes
Unicode-safe image loading via `cv2.imdecode`).

### MangaOCR download SSL errors
If the model is not cached, slow or blocked network can fail. Retry or pre-download
the model. If cached, the warnings can be ignored.

### GPU not used (GGUF)
GGUF needs CUDA-enabled `llama-cpp-python`. Verify GPU support in your environment.

### Missing small bubbles
Increase detection sensitivity or use ComicTextDetector with GPU model if available.

## Development Notes
- Default runtime: `python -m app.main`
- Quick syntax check: `python -m py_compile <files>`
- Keep changes minimal and focused.
- Avoid heavy dependencies unless required.

## License
This project integrates third-party components under their respective licenses.
See `app/third_party/` for details.
