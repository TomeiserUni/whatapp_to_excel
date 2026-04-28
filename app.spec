# app.spec — PyInstaller spec para Windows
# Build: pyinstaller app.spec
# Resultado: dist/WhatsAppExcel/  (pasta, não ficheiro único — mais rápido a arrancar)

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Dados a incluir (modelos, pkl, assets UI)
added_datas = [
    ("data",           "data"),
    ("src/utils.py",   "src"),
    ("src/parser.py",  "src"),
    ("src/pipeline.py","src"),
]
added_datas += collect_data_files("customtkinter")
added_datas += collect_data_files("easyocr")
added_datas += collect_data_files("tkinterdnd2")

hidden = (
    collect_submodules("easyocr") +
    collect_submodules("sentence_transformers") +
    collect_submodules("customtkinter") +
    ["tkinterdnd2", "openpyxl", "rapidfuzz", "PIL", "sklearn"]
)

a = Analysis(
    ["src/app.py"],
    pathex=["."],
    binaries=[],
    datas=added_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "notebook", "IPython", "jupyter"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="WhatsAppExcel",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # sem janela de terminal
    icon=None,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="WhatsAppExcel",
)
