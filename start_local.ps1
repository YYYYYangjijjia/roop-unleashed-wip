param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$appDir = Join-Path $projectRoot 'app'
$uiDir = Join-Path $projectRoot 'react-ui'
$viteEntry = Join-Path $uiDir 'node_modules\vite\bin\vite.js'
$logsDir = Join-Path $projectRoot 'logs'
$statePath = Join-Path $logsDir 'local-services.json'
$uiUrl = 'http://127.0.0.1:5174'
$launchLock = $null
$savedEnv = @{}

function Resolve-LocalPath([string]$value) {
    if ([IO.Path]::IsPathRooted($value)) { return [IO.Path]::GetFullPath($value) }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot $value))
}

$configPath = Join-Path $projectRoot 'runtime.local.json'
$runtime = if (Test-Path -LiteralPath $configPath) {
    Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
} else { $null }
$environment = $runtime.environment
$envMode = if ($environment -and $environment.mode) { [string]$environment.mode } else { 'create' }
if ($envMode -notin @('create', 'existing')) { throw "environment.mode must be create or existing: $configPath" }
$envPath = if ($environment -and $environment.path) { [string]$environment.path } else { 'app/env' }
$envDir = Resolve-LocalPath $envPath
$pythonExe = Join-Path $envDir 'python.exe'
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    $pythonExe = Join-Path $envDir 'Scripts\python.exe'
}
$modelsDir = Resolve-LocalPath $(if ($runtime -and $runtime.models_dir) { [string]$runtime.models_dir } else { 'app/models' })
$offline = [bool]($runtime -and $runtime.offline)
$ffmpegDir = if ($runtime -and $runtime.ffmpeg_dir) {
    Resolve-LocalPath ([string]$runtime.ffmpeg_dir)
} else {
    $ffmpegCommand = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
    if ($ffmpegCommand) { Split-Path -Parent $ffmpegCommand.Source } else { $null }
}

function Get-OwnedProcess($record, [string]$expectedExe) {
    if ($null -eq $record) { return $null }
    $process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if ($null -eq $process) { return $null }
    if ($process.Path -ne $expectedExe -or
        $process.StartTime.ToUniversalTime().ToString('o') -ne $record.startedUtc) {
        throw "Saved PID $($record.pid) now belongs to a different process. No process was stopped."
    }
    return $process
}

function Save-ServiceState {
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$statePath.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$statePath.tmp" -Destination $statePath -Force
}

function Start-LocalService([string]$name, [string]$exe, [string[]]$arguments, [string]$directory) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $stdout = Join-Path $logsDir "$name-$stamp.stdout.log"
    $stderr = Join-Path $logsDir "$name-$stamp.stderr.log"
    $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $directory `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $state.services[$name] = @{
        pid = $process.Id
        startedUtc = $process.StartTime.ToUniversalTime().ToString('o')
        executable = $exe
        stdout = $stdout
        stderr = $stderr
    }
    Save-ServiceState
    return $process
}

function Test-LocalHttp([string]$url) {
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
    } catch { return $false }
}

try {
    if (-not (Test-Path -LiteralPath $modelsDir -PathType Container)) {
        if ($runtime -and $runtime.models_dir) { throw "Configured model directory is missing: $modelsDir" }
        New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
    }
    if (-not $ffmpegDir) { throw 'FFmpeg was not found on PATH; set ffmpeg_dir in runtime.local.json.' }
    foreach ($required in @($pythonExe, $viteEntry, (Join-Path $ffmpegDir 'ffmpeg.exe'))) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file is missing: $required" }
    }
    $nodeExe = (Get-Command node.exe -ErrorAction Stop).Source
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    $launchLock = [IO.File]::Open((Join-Path $logsDir 'local-launch.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    $state = @{ root = $projectRoot; uiUrl = $uiUrl; apiUrl = 'http://127.0.0.1:8002'; gradioUrl = 'http://127.0.0.1:7862'; services = @{} }
    if (Test-Path -LiteralPath $statePath) {
        $oldState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ($oldState.root -ne $projectRoot) { throw "Service state belongs to another project: $statePath" }
        foreach ($name in @('backend', 'frontend')) {
            if ($oldState.services.$name) { $state.services[$name] = $oldState.services.$name }
        }
    }
    $backend = Get-OwnedProcess $state.services.backend $pythonExe
    $frontend = Get-OwnedProcess $state.services.frontend $nodeExe
    foreach ($item in @(@{ port = 8002; process = $backend }, @{ port = 7862; process = $backend }, @{ port = 5174; process = $frontend })) {
        $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $item.port -ErrorAction SilentlyContinue)
        foreach ($listener in $listeners) {
            if ($null -eq $item.process -or $listener.OwningProcess -ne $item.process.Id) {
                throw "Port $($item.port) is occupied by PID $($listener.OwningProcess). No process was stopped."
            }
        }
    }
    $launchEnv = @{
        ROOP_API_PORT = '8002'; ROOP_GRADIO_PORT = '7862'; PORT = '5174'
        ROOP_OFFLINE = $(if ($offline) { '1' } else { '0' })
        HF_HUB_OFFLINE = $(if ($offline) { '1' } else { '0' })
        TRANSFORMERS_OFFLINE = $(if ($offline) { '1' } else { '0' })
        HF_DATASETS_OFFLINE = $(if ($offline) { '1' } else { '0' })
        ROOP_MODELS_DIR = $modelsDir
        HF_HUB_DISABLE_TELEMETRY = '1'; GRADIO_ANALYTICS_ENABLED = 'False'; GRADIO_TELEMETRY_ENABLED = 'False'
        NO_ALBUMENTATIONS_UPDATE = '1'; PYTHONNOUSERSITE = '1'; PYTHONPATH = $null; PYTHONHOME = $null
        PYTHONUTF8 = '1'; ROOP_ADAFACE = '0'
        PATH = "$envDir;$(Join-Path $envDir 'Library\bin');$(Join-Path $envDir 'Scripts');$ffmpegDir;$env:PATH"
    }
    foreach ($key in $launchEnv.Keys) {
        $savedEnv[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $launchEnv[$key], 'Process')
    }
    if ($null -eq $backend) {
        # Both run.py and core.py parse argv; only run.py accepts --execution-provider.
        # CUDA is run.py's default and is also saved in app/config.yaml.
        $backendArgs = @('-u', 'run.py')
        if ($offline) { $backendArgs += '--offline' }
        $backend = Start-LocalService 'backend' $pythonExe $backendArgs $appDir
    }
    if ($null -eq $frontend) {
        $frontend = Start-LocalService 'frontend' $nodeExe @(('"' + $viteEntry + '"'), '--host', '127.0.0.1', '--port', '5174', '--strictPort') $uiDir
    }
    Write-Host 'Waiting for the local backend and browser interface (up to 180 seconds)...'
    $deadline = (Get-Date).AddSeconds(180)
    $ready = $false
    do {
        $backend.Refresh(); $frontend.Refresh()
        if ($backend.HasExited -or $frontend.HasExited) { throw 'A service exited during startup.' }
        if ((Test-LocalHttp "$uiUrl/") -and (Test-LocalHttp "$uiUrl/api/settings") -and
            (Test-LocalHttp 'http://127.0.0.1:7862/')) {
            $ready = $true
            break
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    if (-not $ready) { throw 'Startup timed out. Existing processes were left running for diagnosis.' }
    Save-ServiceState
    Write-Host "C is ready: $uiUrl"
    Write-Host "Service records and logs: $statePath"
    if (-not $NoBrowser) { Start-Process $uiUrl }
} catch {
    Write-Host "Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Logs: $logsDir"
    Write-Host "Service records: $statePath"
    exit 1
} finally {
    foreach ($key in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($key, $savedEnv[$key], 'Process') }
    if ($null -ne $launchLock) { $launchLock.Dispose() }
}
