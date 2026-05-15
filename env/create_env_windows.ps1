param(
    [string]$Python = "python",
    [string]$VenvName = ".venv-latentmas"
)

$ErrorActionPreference = "Stop"

$Workspace = Split-Path -Parent $PSScriptRoot
$Repo = if (Test-Path (Join-Path $Workspace "run.py")) {
    $Workspace
} else {
    Join-Path $Workspace "LatentMAS"
}
$Venv = Join-Path $Workspace $VenvName

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "$Python is not on PATH. Install Python 3.10 first, or pass -Python with the full python.exe path."
}

Invoke-Checked $Python --version
Invoke-Checked $Python -m venv $Venv

$VenvPython = Join-Path $Venv "Scripts\python.exe"
Invoke-Checked $VenvPython -m pip install --upgrade pip
Invoke-Checked $VenvPython -m pip install -r (Join-Path $Repo "requirements.txt")

Write-Host ""
Write-Host "LatentMAS virtual environment is ready."
Write-Host ""
Write-Host "Activate it with:"
Write-Host "  $Venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Verify with:"
Write-Host "  python -m pip show torch transformers datasets accelerate"
Write-Host "  cd `"$Repo`""
Write-Host "  python run.py --help"
