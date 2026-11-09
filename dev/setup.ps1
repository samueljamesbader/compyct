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
    uv python install 3.14
    if (-not (Test-Path packages/InSpice)) {
        git clone https://github.com/samueljamesbader/InSpice packages/InSpice
        if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    }
    if (-not (Test-Path packages/VACASK)) {
        git clone https://codeberg.org/arpadbuermen/VACASK.git packages/VACASK
        if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    }

    # OpenVAF-Reloaded Verilog-A compiler, needed to build VACASK
    $openvafVersion = "osdi_0.5_20260820-7-gde9c640b"
    $openvafUrl = "https://fides.fe.uni-lj.si/openvaf/download/openvaf-reloaded-$openvafVersion-win_x64.zip"
    $openvafDir = Join-Path (Get-Location) ".tools/openvaf"
    if (-not (Test-Path (Join-Path $openvafDir "openvaf-r.exe"))) {
        Write-Host "Downloading OpenVAF-Reloaded $openvafVersion..."
        New-Item -ItemType Directory -Force $openvafDir | Out-Null
        $zip = Join-Path $openvafDir "openvaf.zip"
        Invoke-WebRequest $openvafUrl -OutFile $zip
        Expand-Archive $zip -DestinationPath $openvafDir -Force
        Remove-Item $zip
    }
    $env:SKBUILD_CMAKE_DEFINE = "OPENVAF_DIR=$($openvafDir -replace '\\', '/')"

    uv sync --python 3.14 --extra dev
} finally {
    Pop-Location
}
