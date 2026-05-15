param(
    [string]$Python = "",
    [string]$ModelName = "Qwen/Qwen3-4B",
    [int]$MaxSamples = 1,
    [int]$MaxNewTokens = 512,
    [int]$LatentSteps = 8,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$Workspace = Split-Path -Parent $PSScriptRoot
$Repo = if (Test-Path (Join-Path $Workspace "run.py")) {
    $Workspace
} else {
    Join-Path $Workspace "LatentMAS"
}
$LogDir = Join-Path $Workspace "runs\logs"
$OutDir = Join-Path $Workspace "runs\outputs"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"

New-Item -ItemType Directory -Force -Path $LogDir, $OutDir | Out-Null

$DefaultVenvPython = Join-Path $Workspace ".venv-latentmas\Scripts\python.exe"
if (-not $Python) {
    if (Test-Path $DefaultVenvPython) {
        $Python = $DefaultVenvPython
    } else {
        $Python = "python"
    }
}

$env:HF_HOME = Join-Path $env:USERPROFILE ".cache\huggingface"
$env:TRANSFORMERS_CACHE = $env:HF_HOME
$env:HF_DATASETS_CACHE = $env:HF_HOME

Push-Location $Repo
try {
    & $Python run.py `
        --method latent_mas `
        --model_name $ModelName `
        --task gsm8k `
        --prompt sequential `
        --max_samples $MaxSamples `
        --generate_bs 1 `
        --latent_steps $LatentSteps `
        --max_new_tokens $MaxNewTokens `
        --device $Device 2>&1 |
        Tee-Object -FilePath (Join-Path $LogDir "latentmas_gsm8k_$Stamp.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke test failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
