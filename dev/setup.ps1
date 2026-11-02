# Install uv (if needed) and sync the project environment.
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found; installing..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # Make uv available in this session without restarting
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Push-Location (Join-Path $PSScriptRoot "..")
try {
    if (-not (Test-Path packages/InSpice)) {
        git clone https://github.com/samueljamesbader/InSpice packages/InSpice
        if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    }
    uv sync --extra dev
} finally {
    Pop-Location
}
