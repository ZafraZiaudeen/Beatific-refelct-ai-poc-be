$ErrorActionPreference = 'Stop'

$backendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendRoot = Join-Path (Split-Path -Parent $backendRoot) 'reflect-ai-poc'

Write-Host "Starting Python speaker ID service..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$backendRoot\\python-service'; uvicorn app:app --host 127.0.0.1 --port 8200"

Write-Host "Starting Reflect AI POC backend..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$backendRoot'; npm run dev"

Write-Host "Starting Reflect AI POC frontend..."
Set-Location $frontendRoot
npm run dev
