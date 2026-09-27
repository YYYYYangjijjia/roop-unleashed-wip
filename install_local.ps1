$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$configPath = Join-Path $projectRoot 'runtime.local.json'
$runtime = if (Test-Path -LiteralPath $configPath) {
    Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
} else { $null }
$environment = $runtime.environment
$mode = if ($environment -and $environment.mode) { [string]$environment.mode } else { 'create' }
if ($mode -notin @('create', 'existing')) { throw 'environment.mode must be create or existing.' }
$path = if ($environment -and $environment.path) { [string]$environment.path } else { 'app/env' }
$envDir = if ([IO.Path]::IsPathRooted($path)) { [IO.Path]::GetFullPath($path) }
          else { [IO.Path]::GetFullPath((Join-Path $projectRoot $path)) }
$python = Join-Path $envDir 'python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = Join-Path $envDir 'Scripts\python.exe'
}

if ($mode -eq 'existing') {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "The selected existing Python environment is missing: $envDir"
    }
    & $python -c 'import torch, onnxruntime, cv2, gradio, fastapi'
    if ($LASTEXITCODE -ne 0) { throw 'The existing environment is missing application dependencies; it was not modified.' }
} else {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        if (Test-Path -LiteralPath $envDir) {
            throw "Environment path already exists but contains no Python: $envDir"
        }
        if ($runtime -and $runtime.offline) {
            throw 'Cannot create a new environment while offline is enabled. Use an existing environment or disable offline.'
        }
        & py -3.10 -m venv $envDir
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 environment creation failed. Install Python 3.10 or use Pinokio.' }
        $python = Join-Path $envDir 'Scripts\python.exe'
    }
    & $python -m pip install -r (Join-Path $projectRoot 'app\requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Python requirements installation failed.' }
    $nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($nvidia) {
        & $python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
        if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
        & $python -m pip install filelock fsspec jinja2 networkx typing-extensions sympy onnxruntime-gpu==1.23.2
    } else {
        & $python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
        if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
        & $python -m pip install filelock fsspec jinja2 networkx typing-extensions sympy onnxruntime==1.17.1
    }
    if ($LASTEXITCODE -ne 0) { throw 'ONNX Runtime installation failed.' }
    & $python -m pip install --no-deps sam2 hydra-core omegaconf iopath portalocker antlr4-python3-runtime==4.9.3
    if ($LASTEXITCODE -ne 0) { throw 'Optional SAM2 dependencies installation failed.' }
}

Push-Location (Join-Path $projectRoot 'react-ui')
try {
    & npm ci
    if ($LASTEXITCODE -ne 0) { throw 'React UI dependency installation failed.' }
} finally {
    Pop-Location
}
if ($runtime -and $runtime.optional_models) {
    $unknown = @($runtime.optional_models | Where-Object { $_ -ne 'alphaface' })
    if ($unknown.Count -gt 0) { throw "Unknown optional model(s): $($unknown -join ', ')" }
    if (@($runtime.optional_models) -contains 'alphaface') {
        & $python (Join-Path $projectRoot 'app\tools\install_alphaface.py')
        if ($LASTEXITCODE -ne 0) { throw 'AlphaFace asset installation failed.' }
    }
}
Write-Host "Installation ready. Python: $python"
