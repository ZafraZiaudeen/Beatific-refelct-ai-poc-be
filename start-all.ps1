$ErrorActionPreference = 'Stop'

$backendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendRoot = Join-Path (Split-Path -Parent $backendRoot) 'reflect-ai-poc'
$pythonServiceRoot = Join-Path $backendRoot 'python-service'
$pythonExe = Join-Path $pythonServiceRoot 'venv\Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
  throw "Python virtualenv not found at $pythonExe. Install dependencies in python-service first."
}

Write-Host "Preparing Python speaker ID model..."
& $pythonExe (Join-Path $pythonServiceRoot 'prepare_speaker_model.py')

if ($LASTEXITCODE -ne 0) {
  throw "Python speaker ID model preparation failed."
}

Write-Host "Starting Python speaker ID service..."
$pythonCommand = "Set-Location '$pythonServiceRoot'; & '$pythonExe' -m uvicorn app:app --host 127.0.0.1 --port 8200"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $pythonCommand

Write-Host "Starting Reflect AI POC backend..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$backendRoot'; npm run dev"

Write-Host "Starting Reflect AI POC frontend..."
Set-Location $frontendRoot
npm run dev
