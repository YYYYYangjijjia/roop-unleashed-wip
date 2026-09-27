$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$logsDir = Join-Path $projectRoot 'logs'
$statePath = Join-Path $logsDir 'local-services.json'
$launchLock = $null

function Get-RecordedProcess($record, [string]$expectedExe) {
    if ($null -eq $record) { return $null }
    if ($record.executable -ne $expectedExe -or -not $record.startedUtc -or [int]$record.pid -le 0) {
        throw 'The saved service record does not match this launcher.'
    }
    $process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if ($null -eq $process) { return $null }
    if ($process.Path -ne $expectedExe -or
        $process.StartTime.ToUniversalTime().ToString('o') -ne $record.startedUtc) {
        throw "PID $($record.pid) does not match the recorded executable and start time. Refusing to stop it."
    }
    return $process
}

try {
    if (-not (Test-Path -LiteralPath $logsDir)) {
        Write-Host 'No local service records exist. Nothing was stopped.'
        return
    }
    $launchLock = [IO.File]::Open((Join-Path $logsDir 'local-launch.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    if (-not (Test-Path -LiteralPath $statePath)) {
        Write-Host 'No local service records exist. Nothing was stopped.'
        return
    }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.root -ne $projectRoot) { throw "Service state belongs to another project: $statePath" }
    $expected = @{
        backend = [string]$state.services.backend.executable
        frontend = [string]$state.services.frontend.executable
    }
    # Validate both identities before stopping either service. Reused PIDs fail closed.
    foreach ($name in @('frontend', 'backend')) {
        $null = Get-RecordedProcess $state.services.$name $expected[$name]
    }
    foreach ($name in @('frontend', 'backend')) {
        $process = Get-RecordedProcess $state.services.$name $expected[$name]
        if ($null -ne $process) {
            Stop-Process -InputObject $process -ErrorAction Stop
            if (-not $process.WaitForExit(10000)) { throw "Timed out waiting for $name PID $($process.Id) to exit." }
            Write-Host "Stopped C $name (PID $($process.Id))."
        }
    }
    $state.services = @{}
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$statePath.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$statePath.tmp" -Destination $statePath -Force
    Write-Host 'C local services are stopped. Logs have been preserved.'
} catch {
    Write-Host "Stop failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Service records: $statePath"
    exit 1
} finally {
    if ($null -ne $launchLock) { $launchLock.Dispose() }
}
