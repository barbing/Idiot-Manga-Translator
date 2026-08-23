# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

brand_asset_dir = Path(SPECPATH).resolve() / 'app' / 'assets' / 'branding'
brand_asset_names = ('yomiframe.ico', 'yomiframe-1024.png')
missing_brand_assets = [
    name for name in brand_asset_names if not (brand_asset_dir / name).is_file()
]
if missing_brand_assets:
    raise RuntimeError(
        'Missing required YomiFrame brand assets: '
        + ', '.join(missing_brand_assets)
    )
for asset_name in brand_asset_names:
    datas.append((str(brand_asset_dir / asset_name), 'app/assets/branding'))

brand_icon_path = brand_asset_dir / 'yomiframe.ico'

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

# Add app package explicitly
hiddenimports += collect_submodules('app')

block_cipher = None

a = Analysis(
    ['app/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[], 
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='YomiFrame_Single',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    icon=str(brand_icon_path),
    codesign_identity=None,
    entitlements_file=None,
)
