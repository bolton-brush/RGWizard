# -*- mode: python ; coding: utf-8 -*-

import sys
import runpy
from pathlib import Path


metadata = runpy.run_path(Path(SPECPATH) / 'app_metadata.py')
APP_NAME = metadata['APP_NAME']
APP_VERSION = metadata['APP_VERSION']


is_windows = sys.platform == 'win32'
optimize_binaries = not is_windows

datas = []
if Path('./models').exists():
    datas.append(('./models', 'models'))

hiddenimports = [
    'PIL._tkinter_finder',
    'imagecodecs._shared',
    'imagecodecs._imcd',
    '69e79323355fec38eab6__mypyc',
    'numpy._core.multiarray',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=optimize_binaries,
    upx=optimize_binaries,
    upx_exclude=[],
    runtime_tmpdir=None,
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
    a.datas,
    strip=optimize_binaries,
    upx=optimize_binaries,
    upx_exclude=[],
    name=APP_NAME,
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name=f'{APP_NAME}.app',
        bundle_identifier='org.frenchfaso.rgwizard',
        info_plist={
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': APP_NAME,
            'CFBundleShortVersionString': APP_VERSION,
            'CFBundleVersion': APP_VERSION,
            'NSHighResolutionCapable': True,
        },
    )
