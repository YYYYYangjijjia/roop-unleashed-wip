$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$configPath = Join-Path $projectRoot 'runtime.local.json'
$examplePath = Join-Path $projectRoot 'runtime.example.json'
if (-not (Test-Path -LiteralPath $configPath)) {
    Copy-Item -LiteralPath $examplePath -Destination $configPath
    Write-Host "Created runtime.local.json from runtime.example.json; edit it for this machine if needed."
}
$runtime = if (Test-Path -LiteralPath $configPath) {
    Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
} else { $null }
$modelProfile = if ($runtime -and $runtime.model_profile) { [string]$runtime.model_profile } else { 'local_default' }
if ($modelProfile -notin @('local_default', 'none')) { throw "Unknown model_profile: $modelProfile" }
$settingsPath = Join-Path $projectRoot 'app\config.yaml'
$settingsTemplate = Join-Path $projectRoot 'app\config.default.yaml'
if (-not (Test-Path -LiteralPath $settingsPath)) {
    Copy-Item -LiteralPath $settingsTemplate -Destination $settingsPath
    Write-Host 'Created app/config.yaml from the portable default settings.'
}
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
    $nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($nvidia) {
        Write-Host 'Checking PyTorch 2.7.0 CUDA 12.8; matching installed wheels will be reused.'
        & $python -m pip install 'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' --index-url https://download.pytorch.org/whl/cu128 --no-deps
        if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
        & $python -m pip install filelock fsspec jinja2 networkx typing-extensions sympy onnxruntime-gpu==1.23.2
    } else {
        Write-Host 'Checking PyTorch 2.7.0 CPU; matching installed wheels will be reused.'
        & $python -m pip install 'torch==2.7.0+cpu' 'torchvision==0.22.0+cpu' --index-url https://download.pytorch.org/whl/cpu --no-deps
        if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
        & $python -m pip install filelock fsspec jinja2 networkx typing-extensions sympy onnxruntime==1.17.1
    }
    if ($LASTEXITCODE -ne 0) { throw 'ONNX Runtime installation failed.' }
    & $python -m pip install -r (Join-Path $projectRoot 'app\requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Python requirements installation failed.' }
    & $python -m pip install --no-deps sam2 hydra-core omegaconf iopath portalocker antlr4-python3-runtime==4.9.3
    if ($LASTEXITCODE -ne 0) { throw 'Optional SAM2 dependencies installation failed.' }
}

if ($modelProfile -eq 'local_default') {
    if ($mode -eq 'create') {
        # MediaPipe Tasks powers the default UV detail pipeline. Installing its
        # wheel without dependencies preserves the one headless OpenCV build.
        & $python -m pip install 'absl-py==2.3.1' 'flatbuffers==24.3.25' 'matplotlib==3.9.1' 'sounddevice==0.5.2'
        if ($LASTEXITCODE -ne 0) { throw 'MediaPipe support dependencies installation failed.' }
        & $python -m pip install --no-deps 'mediapipe==1.0.1'
        if ($LASTEXITCODE -ne 0) { throw 'MediaPipe installation failed.' }
    }
    & $python -c 'import mediapipe as mp; assert hasattr(mp.tasks.vision, "FaceLandmarker")'
    if ($LASTEXITCODE -ne 0) { throw 'MediaPipe Tasks is missing from this Python environment. Install it before using the default UV identity-detail setting.' }
    & $python (Join-Path $projectRoot 'app\tools\install_default_models.py')
    if ($LASTEXITCODE -ne 0) { throw 'Default model installation or verification failed.' }
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
