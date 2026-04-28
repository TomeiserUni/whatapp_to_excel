# build/app.spec — PyInstaller onefile para Windows
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = [
    ("../data",                   "data"),
    ("../src/utils.py",           "src"),
    ("../src/parser.py",          "src"),
    ("../src/pipeline.py",        "src"),
    ("../src/templates",          "src/templates"),
]
datas += collect_data_files("easyocr")
datas += collect_data_files("flask")

hiddenimports = (
    collect_submodules("easyocr") +
    collect_submodules("sentence_transformers") +
    collect_submodules("flask") +
    ["openpyxl", "rapidfuzz", "PIL", "sklearn"]
)

a = Analysis(
    ["../src/server.py"],
    pathex=[".."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "notebook", "IPython", "jupyter"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
