$ErrorActionPreference = "Continue"

Write-Host "Initializing Pipeline Environment..." -ForegroundColor Cyan

# Point to the OneAPI setvars.bat, which automatically configures 
# INCLUDE, LIB, and PATH for the compiler.
$setvarsPath = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if (Test-Path $setvarsPath) {
    cmd.exe /c "`"$setvarsPath`" && set" | ForEach-Object {
        $key, $value = $_ -split '=', 2
        if ($key -and $value) {
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    Write-Host "Intel OneAPI environment variables set." -ForegroundColor Green
}

# 1. Clean the local PATH to prevent length overflow errors
$env:PATH = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

# 2. Force HF Cache and load token
$env:HF_HOME = "F:\JEPA_Model\hf_cache"
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#\s]+)\s*=\s*(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
    Write-Host "Hugging Face Authentication Token Loaded." -ForegroundColor Green
}

# 3. Explicitly link the Intel oneAPI SYCL Compiler to Triton
# We use icpx.exe (Intel C++) because it natively handles SYCL and Windows /LIBPATH flags
$intelCompilerPath = "C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin\compiler"
$env:PATH = "$intelCompilerPath;" + $env:PATH
# Force Python/Triton to use the Intel SYCL Compiler
$env:CC = "icx"
$env:CXX = "icpx"

# Optional safety net to strip aggressive Windows linker formats if they persist
$env:LDFLAGS = " " 

Write-Host "Triton SYCL Compiler forcefully mapped to Intel oneAPI: $env:CXX" -ForegroundColor Green

# Check for Admin rights
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "WARNING: Script is not running as Administrator. Some operations may fail." -ForegroundColor Yellow
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
Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] `nStarting Stage 0: Pre-flight Cache..." -ForegroundColor Cyan
$startTime = Get-Date
python src/download_models.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 0 failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}
$duration = (Get-Date) - $startTime
Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] Stage 0 Completed in $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Green

# Stage 1
Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] `nStarting Stage 1: Dataset Extraction & Formatting..." -ForegroundColor Cyan
$startTime = Get-Date
python src/extract_frontier_data.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 1 failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}
$duration = (Get-Date) - $startTime
Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] Stage 1 Completed in $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Green

# Stage 3
Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] \nStarting Stage 3A: JEPA Loop Training (Frontier Traces)..." -ForegroundColor Cyan
$startTime = Get-Date
python src/train_latent_loop.py --epochs 2 --curriculum_phase "frontier_traces"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 3A failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}
$duration = (Get-Date) - $startTime
Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] Stage 3A Completed in $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Green

Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] \nStarting Stage 3B: JEPA Loop Training (General Knowledge)..." -ForegroundColor Cyan
$startTime = Get-Date
python src/train_latent_loop.py --epochs 2 --curriculum_phase "general_knowledge"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 3B failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}
$duration = (Get-Date) - $startTime
Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] Stage 3B Completed in $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Green

Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] \nStarting Stage 3C: JEPA Loop Training (Code Mechanics)..." -ForegroundColor Cyan
$startTime = Get-Date
python src/train_latent_loop.py --epochs 2 --curriculum_phase "code_mechanics"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 3C failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}
$duration = (Get-Date) - $startTime
Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] Stage 3C Completed in $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Green

# # Stage 4
# Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] `nStarting Stage 4: Decoder Training..." -ForegroundColor Cyan
# $startTime = Get-Date
# python src/train_decoder.py
# if ($LASTEXITCODE -ne 0) {
#     Write-Host "ERROR: Stage 4 failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
#     [System.Environment]::Exit($LASTEXITCODE)
# }

# Stage 5
Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] `nStarting Stage 5: Inference Harness..." -ForegroundColor Cyan
$startTime = Get-Date
python src/inference_harness.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stage 5 failed with exit code $LASTEXITCODE. Halting pipeline to prevent cascading errors." -ForegroundColor Red
    [System.Environment]::Exit($LASTEXITCODE)
}
$duration = (Get-Date) - $startTime
Write-Host "[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] Stage 5 Completed in $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Green

Write-Host "`n[$([Get-Date -Format 'yyyy-MM-dd HH:mm:ss'])] Pipeline completed successfully! All stages finished without errors." -ForegroundColor Green
