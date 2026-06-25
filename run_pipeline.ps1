$ErrorActionPreference = "Continue"

Write-Host "Initializing Pipeline Environment..." -ForegroundColor Cyan

# Check for Admin rights
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "WARNING: Script is not running as Administrator. Some operations may fail." -ForegroundColor Yellow
} else {
    Write-Host "Confirmed: Running as Administrator." -ForegroundColor Green
}

# Activate virtual environment
$venvPath = ".\venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment at $venvPath" -ForegroundColor Green
    . $venvPath
} else {
    Write-Host "ERROR: Virtual environment not found at $venvPath. Please ensure the environment is set up." -ForegroundColor Red
    [System.Environment]::Exit(1)
}

# Stage 0
Write-Host "`nStarting Stage 0: Pre-flight Cache..." -ForegroundColor Cyan
python src/download_models.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 0 failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}

# Stage 1
Write-Host "`nStarting Stage 1: Dataset Preparation..." -ForegroundColor Cyan
python src/dataset_preparation.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 1 failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}

# Stage 2
Write-Host "`nStarting Stage 2: Teacher Distillation..." -ForegroundColor Cyan
python src/teacher_distillation.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 2 failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}

# Stage 3
Write-Host "`nStarting Stage 3: JEPA Loop Training..." -ForegroundColor Cyan
python src/train_latent_loop.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 3 failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}

# Stage 4
Write-Host "`nStarting Stage 4: Decoder Training..." -ForegroundColor Cyan
python src/train_decoder.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 4 failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}

# Stage 5
Write-Host "`nStarting Stage 5: Inference Harness..." -ForegroundColor Cyan
python src/inference_harness.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 5 failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}

Write-Host "`nPipeline completed successfully! All stages finished without errors." -ForegroundColor Green
