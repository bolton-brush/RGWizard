# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path


datas = []
if Path('./models').exists():
    datas.append(('./models', 'models'))


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
    a.binaries,
    a.datas,
    [],
    name='RGWizard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='RGWizard.app',
        bundle_identifier='org.frenchfaso.rgwizard',
        info_plist={
            'CFBundleName': 'RGWizard',
            'CFBundleDisplayName': 'RGWizard',
            'NSHighResolutionCapable': True,
        },
    )
