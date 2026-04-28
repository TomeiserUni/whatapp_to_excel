# instalar.ps1 — Executa como Administrador na primeira vez
# Instala Python embeddable + dependências e cria atalho no Ambiente de Trabalho

param([string]$PythonVer = "3.11.9")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot   # pasta raiz do projecto

Write-Host "=== WhatsApp → Excel — Instalador ===" -ForegroundColor Cyan

# ── 1. Descarregar Python embeddable ─────────────────────────────────────────
$PyDir  = "$Root\windows\python"
$PyExe  = "$PyDir\python.exe"
$PipExe = "$PyDir\Scripts\pip.exe"

if (-not (Test-Path $PyExe)) {
    Write-Host "A descarregar Python $PythonVer (embeddable)..." -ForegroundColor Yellow
    $Url  = "https://www.python.org/ftp/python/$PythonVer/python-$PythonVer-embed-amd64.zip"
    $Zip  = "$env:TEMP\python-embed.zip"
    Invoke-WebRequest $Url -OutFile $Zip
    Expand-Archive $Zip -DestinationPath $PyDir -Force
    Remove-Item $Zip

    # Activar import de site-packages no Python embeddable
    $Pth = Get-ChildItem "$PyDir\*._pth" | Select-Object -First 1
    (Get-Content $Pth.FullName) -replace "#import site", "import site" |
        Set-Content $Pth.FullName

    # Instalar pip
    Write-Host "A instalar pip..." -ForegroundColor Yellow
    Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile "$env:TEMP\get-pip.py"
    & $PyExe "$env:TEMP\get-pip.py" --no-warn-script-location
}

# ── 2. Instalar PyTorch CPU ───────────────────────────────────────────────────
Write-Host "A instalar PyTorch (CPU)..." -ForegroundColor Yellow
& $PipExe install --no-warn-script-location `
    torch==2.0.1+cpu `
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── 3. Instalar restantes dependências ───────────────────────────────────────
Write-Host "A instalar dependências..." -ForegroundColor Yellow
& $PipExe install --no-warn-script-location -r "$Root\requirements.txt"

# ── 4. Pré-descarregar modelos EasyOCR ───────────────────────────────────────
Write-Host "A descarregar modelos OCR (~300 MB, só na primeira vez)..." -ForegroundColor Yellow
& $PyExe -c "import easyocr; easyocr.Reader(['pt'], gpu=False)" 2>&1 | Write-Host

# ── 5. Criar atalho no Ambiente de Trabalho ───────────────────────────────────
$Shell   = New-Object -ComObject WScript.Shell
$Desktop = $Shell.SpecialFolders("Desktop")
$Link    = $Shell.CreateShortcut("$Desktop\WhatsApp Excel.lnk")
$Link.TargetPath       = $PyExe
$Link.Arguments        = "`"$Root\src\app.py`""
$Link.WorkingDirectory = $Root
$Link.IconLocation     = "shell32.dll,21"
$Link.Save()

Write-Host ""
Write-Host "Instalacao concluida!" -ForegroundColor Green
Write-Host "Usa o atalho 'WhatsApp Excel' no Ambiente de Trabalho." -ForegroundColor Green
Write-Host "Ou corre manualmente:  windows\python\python.exe src\app.py"
