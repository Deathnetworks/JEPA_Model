param (
    [string]$ConfigPath = "configs/default.yaml",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

Write-Host "Starting JEPA_Model v2 Pipeline Orchestration" -ForegroundColor Green

if (Test-Path "venv\\Scripts\\Activate.ps1") {
    . "venv\\Scripts\\Activate.ps1"
}

Write-Host "Executing Multi-Stage Training Pipeline..." -ForegroundColor Yellow

$Command = "python src/train_pipeline.py --config $ConfigPath"
if ($Resume) {
    $Command += " --resume"
}

Invoke-Expression $Command

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline execution failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Pipeline execution completed successfully." -ForegroundColor Green
