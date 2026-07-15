# RAG Q&A Assistant - Build & Run Script
# Cost: $0 (local embeddings + Groq free tier)

$ErrorActionPreference = "Stop"
$projDir = $PSScriptRoot
$venvPython = "$projDir\venv\Scripts\python.exe"
$venvPip = "$projDir\venv\Scripts\pip.exe"

Write-Host "`n=== RAG Q&A Assistant Setup ===" -ForegroundColor Cyan

# Step 1: Create venv
if (-not (Test-Path $venvPython)) {
    Write-Host "`n[1/5] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv "$projDir\venv"
} else {
    Write-Host "`n[1/5] Virtual environment already exists." -ForegroundColor Yellow
}

# Step 2: Verify venv exists
Write-Host "[2/5] Verifying venv..." -ForegroundColor Yellow
if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: venv creation failed." -ForegroundColor Red
    exit 1
}
Write-Host "  venv OK: $venvPython" -ForegroundColor Green

# Step 3: Install dependencies using venv pip directly
Write-Host "[3/5] Installing dependencies..." -ForegroundColor Yellow
& $venvPip install -r "$projDir\requirements.txt" --quiet

# Step 4: Copy .env.example to .env
if (-not (Test-Path "$projDir\.env")) {
    Write-Host "[4/5] Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item "$projDir\.env.example" "$projDir\.env"
} else {
    Write-Host "[4/5] .env already exists, skipping." -ForegroundColor Yellow
}

# Step 5: Syntax check
Write-Host "[5/5] Running syntax check..." -ForegroundColor Yellow
$pyFiles = @()
$pyFiles += Get-ChildItem "$projDir\src\*.py" -ErrorAction SilentlyContinue
$pyFiles += Get-ChildItem "$projDir\app.py" -ErrorAction SilentlyContinue
$pyFiles += Get-ChildItem "$projDir\evaluation\*.py" -ErrorAction SilentlyContinue

foreach ($f in $pyFiles) {
    & $venvPython -m py_compile $f.FullName
}

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "Launching Streamlit app...`n" -ForegroundColor Cyan
& $venvPython -m streamlit run "$projDir\app.py"
