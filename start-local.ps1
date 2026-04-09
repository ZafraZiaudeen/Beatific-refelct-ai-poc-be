$ErrorActionPreference = 'Stop'

$pythonServiceRoot = Join-Path $PSScriptRoot 'python-service'
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
Set-Location $PSScriptRoot
npm run dev
