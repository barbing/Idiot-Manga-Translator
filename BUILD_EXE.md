# Building YomiFrame EXE

## Prerequisites

1. **Use the existing Conda environment**:

    ```powershell
    conda activate manga-llm
    ```

2. **Install the pinned self-contained PyICU runtime for packaging**:

    ```powershell
    python -m pip install --no-deps --force-reinstall "https://github.com/barbing/YomiFrame-LLM_Manga_Translator/releases/download/runtime-dependencies-v1/pyicu-2.16.2-cp310-cp310-win_amd64.whl#sha256=a20721fe04dcfd8b34c17e2f45ba45beebaf32f1a03bd07efc07a512c2b3f830"
    ```

    This project-hosted CPython 3.10 x64 wheel contains PyICU 2.16.2, ICU4C
    78.3, and the three application-private ICU DLLs required by the frozen
    application. The hash fragment makes `pip` reject any bytes that differ
    from the accepted release artifact.

    This step is for maintainers producing a frozen build. Ordinary Windows
    source launches obtain the same SHA-256-pinned wheel through the existing
    GUI downloader and install it below `%LOCALAPPDATA%\YomiFrame`. The
    deployment script under `scripts/deployment` is only for creating a future
    release-catalog wheel; it is not a routine EXE-build prerequisite.

3. **Install PyInstaller**:

    ```powershell
    pip install pyinstaller
    ```

4. **Ensure the remaining dependencies are installed**:

    ```powershell
    pip install -r requirements.txt
    ```

## Build Instructions
Run the following command in the terminal:

```powershell
pyinstaller manga_translator.spec
```

## Output
*   The built application will be in `dist/YomiFrame`.
*   Run `YomiFrame.exe` inside that folder.

Both PyInstaller specifications embed `app/assets/branding/yomiframe.ico` in
the executable and collect the same canonical icon plus the reviewed 1024 px
raster master for Qt runtime use. Keep the executable resource and runtime
brand assets together; replacing only one of them causes Windows shell and
in-app branding to diverge.

The onedir specification fails closed at build time when the pinned PyICU
package, its wheel-adjacent `_icu_*.pyd`, exact `icudt78.dll`, `icuin78.dll`,
`icuuc78.dll`, or the required license notices are absent. It collects that
self-contained package without consulting Conda `Library/bin`, system ICU, or
Qt ICU, and excludes its native files from UPX.

Before publishing a source release, upload the exact self-contained wheel at
the immutable release URL embedded in `app/models/downloader.py` and verify its
SHA-256 against the embedded pin. The wheel is a release artifact and must not
be committed to the repository. Packaged builds use their bundled runtime and
do not consult that URL.

Deployment assets are organized under the public, versioned
`runtime-dependencies-v1` release. The current immutable asset is:

- URL: `https://github.com/barbing/YomiFrame-LLM_Manga_Translator/releases/download/runtime-dependencies-v1/pyicu-2.16.2-cp310-cp310-win_amd64.whl`
- SHA-256: `a20721fe04dcfd8b34c17e2f45ba45beebaf32f1a03bd07efc07a512c2b3f830`

Never replace an existing asset in this catalog. Compatible future deployment
files may be added under v1; incompatible catalog changes require a new
versioned runtime-dependencies release.

## Models
The EXE does **not** bundle the large model files (to prevent the EXE from being 10GB+).
*   **Copy your `models/` folder** into the `dist/YomiFrame/` folder.
*   The first time you run it on a new machine, it may try to download required OCR, detection, or cleanup inpainting assets if they are not cached.
    *   To make it truly offline, prepare the required model assets and caches on the target machine before launching the packaged app.
