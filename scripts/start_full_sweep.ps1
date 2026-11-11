$ErrorActionPreference = "Stop"

$projectRoot = Split-Path $PSScriptRoot -Parent
$workerScript = Join-Path $PSScriptRoot "run_full_sweep_background.ps1"
$pidFile = Join-Path $projectRoot "results\izh-full-sweep-utf8-background.pid"

if (Test-Path -LiteralPath $pidFile) {
    $oldPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
        Write-Output "The full izh sweep is already running with PID $oldPid."
        exit 0
    }
}

$process = Start-Process powershell.exe `
    -WindowStyle Hidden `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $workerScript
    ) `
    -PassThru

$process.Id | Set-Content -LiteralPath $pidFile -Encoding ASCII
Write-Output "Started the full izh sweep in the background with PID $($process.Id)."
Write-Output "Check progress with: python scripts/sweep_status.py"
