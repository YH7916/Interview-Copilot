$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python not found: $python"
}

Set-Location $root
& $python -m nanobot.cli.commands interview-agent
