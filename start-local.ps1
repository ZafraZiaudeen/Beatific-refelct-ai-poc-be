$ErrorActionPreference = 'Stop'

Write-Host "Starting Python speaker ID service..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot\\python-service'; uvicorn app:app --host 127.0.0.1 --port 8200"

Write-Host "Starting Reflect AI POC backend..."
Set-Location $PSScriptRoot
npm run dev
