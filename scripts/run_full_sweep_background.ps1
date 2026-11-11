$ErrorActionPreference = "Stop"

$projectRoot = Split-Path $PSScriptRoot -Parent
$python = "D:\ai\.miniconda\envs\fst-distillation\python.exe"
$results = Join-Path $projectRoot "results"
$log = Join-Path $results "izh-full-sweep-utf8-background.log"
$errorLog = Join-Path $results "izh-full-sweep-utf8-background-error.log"
$status = Join-Path $results "izh-full-sweep-utf8-background-status.json"

Set-Location $projectRoot
New-Item -ItemType Directory -Force -Path $results | Out-Null

@{
    state = "running"
    pid = $PID
    started_at = (Get-Date).ToString("o")
    command = "python -m scripts.full_local_sweep --models-folder models/izh-full-sweep-utf8 --results-folder results/izh-full-sweep-utf8 --storage sqlite:///results/izh-full-sweep-utf8.db"
} | ConvertTo-Json | Set-Content -LiteralPath $status -Encoding UTF8

try {
    $env:PYTHONUNBUFFERED = "1"
    $pythonProcess = Start-Process $python `
        -ArgumentList @(
            "-m", "scripts.full_local_sweep",
            "--models-folder", "models/izh-full-sweep-utf8",
            "--results-folder", "results/izh-full-sweep-utf8",
            "--storage", "sqlite:///results/izh-full-sweep-utf8.db"
        ) `
        -RedirectStandardOutput $log `
        -RedirectStandardError $errorLog `
        -NoNewWindow `
        -Wait `
        -PassThru
    $exitCode = $pythonProcess.ExitCode
    $state = if ($exitCode -eq 0) { "completed" } else { "failed" }
    @{
        state = $state
        pid = $PID
        exit_code = $exitCode
        finished_at = (Get-Date).ToString("o")
        log = $log
        error_log = $errorLog
    } | ConvertTo-Json | Set-Content -LiteralPath $status -Encoding UTF8
    exit $exitCode
}
catch {
    @{
        state = "failed"
        pid = $PID
        exit_code = 1
        finished_at = (Get-Date).ToString("o")
        error = $_.Exception.Message
        log = $log
        error_log = $errorLog
    } | ConvertTo-Json | Set-Content -LiteralPath $status -Encoding UTF8
    exit 1
}
