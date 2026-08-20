# -*- mode: python ; coding: utf-8 -*-
import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# Collect packages that might be missed
packages = [
    'manga_ocr', 
    'simple_lama_inpainting',
    'imghdr', 
    'requests',
    'PIL',
    'cv2',
]

for package in packages:
    try:
        tmp_ret = collect_all(package)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception:
        continue

# PyICU's accepted Windows wheel is a self-contained package. Its extension and
# ICU libraries must remain adjacent in the installed package; never substitute
# Conda, system, or Qt ICU binaries.
icu_spec = importlib.util.find_spec('icu')
if icu_spec is None or not icu_spec.origin:
    raise RuntimeError(
        'The pinned self-contained PyICU wheel is required for packaging. '
        'Install the hash-bound release artifact documented in BUILD_EXE.md.'
    )

icu_package_dir = Path(icu_spec.origin).resolve().parent
icu_extension = icu_package_dir / '_icu_.cp310-win_amd64.pyd'
required_icu_runtime_names = ('icudt78.dll', 'icuin78.dll', 'icuuc78.dll')
missing_icu_runtime = [
    name for name in required_icu_runtime_names
    if not (icu_package_dir / name).is_file()
]
if not icu_extension.is_file() or missing_icu_runtime:
    missing = list(missing_icu_runtime)
    if not icu_extension.is_file():
        missing.insert(0, icu_extension.name)
    raise RuntimeError(
        'The installed PyICU package is not the complete self-contained '
        f'runtime documented in BUILD_EXE.md; missing: {", ".join(missing)}'
    )

icu_data, icu_binaries, icu_hiddenimports = collect_all('icu')
# collect_all classifies the wheel DLLs as data as well as binaries and also
# collects the wheel's internal ICU notice. Keep one binary copy beside the
# extension and use the project's deployment notices as the single packaged
# license location.
icu_runtime_data_names = {
    *(name.lower() for name in required_icu_runtime_names),
    'icu-license.txt',
}
icu_data = [
    entry for entry in icu_data
    if Path(entry[0]).name.lower() not in icu_runtime_data_names
]
datas += icu_data
binaries += icu_binaries
hiddenimports += icu_hiddenimports

license_dir = Path(SPECPATH).resolve() / 'licenses'
for license_name in ('ICU-LICENSE.txt', 'PyICU-LICENSE.txt'):
    license_path = license_dir / license_name
    if not license_path.is_file():
        raise RuntimeError(f'Missing required deployment notice: {license_path}')
    datas.append((str(license_path), 'licenses'))

# Add app package explicitly
hiddenimports += collect_submodules('app')

block_cipher = None
icu_upx_excludes = [
    'icu.pyd',
    'icu*.pyd',
    'icudt*.dll',
    'icuin*.dll',
    'icuuc*.dll',
]

a = Analysis(
    ['app/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # We do not exclude torch because we want a standalone (folder) distribution.
    excludes=[], 
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# PyInstaller's dependency scan also flattens the wheel-owned ICU DLLs and can
# discover Conda's unversioned forwarding aliases. The package-local copies are
# the authoritative runtime used by icu._icu_; remove only redundant flat ICU
# entries from the final binary table.
flat_icu_runtime_names = {
    *(name.lower() for name in required_icu_runtime_names),
    'icudt.dll',
    'icuin.dll',
    'icuuc.dll',
}
a.binaries = [
    entry for entry in a.binaries
    if not (
        '/' not in str(entry[0]).replace('\\', '/')
        and str(entry[0]).lower() in flat_icu_runtime_names
    )
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YomiFrame',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=icu_upx_excludes,
    name='YomiFrame',
)
