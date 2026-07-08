# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path


datas = []
if Path('./models').exists():
    datas.append(('./models', 'models'))

strip_binaries = sys.platform.startswith('linux')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['PIL._tkinter_finder', 'imagecodecs._shared', 'imagecodecs._imcd'],
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
    name='RGWizard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=strip_binaries,
    upx=False,
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
    strip=strip_binaries,
    upx=False,
    upx_exclude=[],
    name='RGWizard',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='RGWizard.app',
        bundle_identifier='org.frenchfaso.rgwizard',
        info_plist={
            'CFBundleName': 'RGWizard',
            'CFBundleDisplayName': 'RGWizard',
            'NSHighResolutionCapable': True,
        },
    )
