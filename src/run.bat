@echo off
SETLOCAL EnableDelayedExpansion
title JEPA Local Vector Extraction Pipeline Engine

echo =====================================================================
echo 1. INITIALIZING INTEL oneAPI XPU RUNTIME VARIABLES
echo =====================================================================

:: Define common Intel oneAPI installation paths for Windows
set "INTEL_SETVARS_1=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
set "INTEL_SETVARS_2=C:\Program Files\Intel\oneAPI\setvars.bat"

if exist "%INTEL_SETVARS_1%" (
    echo Found Intel oneAPI environment variables path configuration. Loading...
    call "%INTEL_SETVARS_1%" status
) else if exist "%INTEL_SETVARS_2%" (
    echo Found Intel oneAPI environment variables path configuration. Loading...
    call "%INTEL_SETVARS_2%" status
) else (
    echo [WARNING] setvars.bat not found automatically in standard locations.
    echo If IPEX framework initialization errors occur, please verify your Intel oneAPI Toolkit installation path.
)

:: Set standard Intel runtime environment performance optimization keys
set OMP_NUM_THREADS=4
set MKL_DYNAMIC=TRUE

echo.
echo =====================================================================
echo 2. MANAGING PYTHON VIRTUAL ENVIRONMENT (VENV)
echo =====================================================================

if not exist "venv\" (
    echo Local directory 'venv' directory not found. Spawning clean virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [CRITICAL ERROR] Failed to create Python virtual environment. Verify Python is installed and added to PATH.
        pause
        exit /b
    )
    echo Virtual environment created successfully.
) else (
    echo Existing local virtual environment verified.
)

echo Activating environment context...
call venv\Scripts\activate

echo.
echo =====================================================================
echo 3. VERIFYING PIP DEPENDENCY INFRASTRUCTURE
echo =====================================================================
echo Checking required libraries...

:: Clean background confirmation sweep to prevent redundant download latency checks
python -c "import transformers, datasets, accelerate, bitsandbytes, teich, torchcodec" >nul 2>&1

if errorlevel 1 (
    echo Elements missing or unverified. Launching explicit dependency updates...
    pip install --disable-pip-version-check -q transformers datasets accelerate bitsandbytes teich torchcodec
    pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu --index-url https://download.pytorch.org/whl/xpu
    
    echo.
    echo Note: If you have not done so already, ensure your Intel XPU-compatible 
    echo PyTorch and IPEX packages are installed in this venv via Intel's channels:
    echo e.g., pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu --index-url https://download.pytorch.org/whl/xpu
) else (
    echo All processing elements verified successfully.
)

echo.
echo =====================================================================
echo 4. EXECUTING LOCAL WORKSTATION VECTOR PIPELINE
echo =====================================================================
echo Control handed off to python runtime engine...
echo.

python extract_frontier_data.py --chunk_size 10000

echo.
echo =====================================================================
echo PIPELINE TERMINATED OR CONCLUDED EXHAUSTED QUEUES
echo =====================================================================
pause