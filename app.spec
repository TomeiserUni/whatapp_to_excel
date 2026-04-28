# app.spec — PyInstaller onefile para Windows
# Build automático: faz push com tag   git tag v1.0 && git push --tags
# O .exe aparece nos artefactos do GitHub Actions

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = [
    ("data",            "data"),
    ("src/utils.py",    "src"),
    ("src/parser.py",   "src"),
    ("src/pipeline.py", "src"),
]
datas += collect_data_files("customtkinter")
datas += collect_data_files("easyocr")
datas += collect_data_files("tkinterdnd2")

hiddenimports = (
    collect_submodules("easyocr") +
    collect_submodules("sentence_transformers") +
    collect_submodules("customtkinter") +
    ["tkinterdnd2", "openpyxl", "rapidfuzz", "PIL", "sklearn"]
)

a = Analysis(
    ["src/app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "notebook", "IPython", "jupyter"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Tudo num único .exe  (extrai para %TEMP% ao arrancar — demora ~15s na primeira vez)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="WhatsAppExcel",
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)
