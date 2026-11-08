param(
    [ValidateSet("czn", "izh")]
    [string[]]$Datasets = @("czn", "izh")
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path $repositoryRoot "data\inflection"

$developmentFolders = @{
    "czn" = "DEVELOPMENT-LANGUAGES\oto-manguean"
    "izh" = "DEVELOPMENT-LANGUAGES\uralic"
}

if (-not (Test-Path (Join-Path $dataRoot "DEVELOPMENT-LANGUAGES"))) {
    throw "SIGMORPHON data submodule is missing. Run: git submodule update --init --recursive"
}

foreach ($dataset in $Datasets) {
    $developmentFolder = Join-Path $dataRoot $developmentFolders[$dataset]
    $goldTestFolder = Join-Path $dataRoot "GOLD-TEST"

    Copy-Item -LiteralPath (Join-Path $developmentFolder "$dataset.trn") `
        -Destination (Join-Path $dataRoot "$dataset.trn") -Force
    Copy-Item -LiteralPath (Join-Path $developmentFolder "$dataset.dev") `
        -Destination (Join-Path $dataRoot "$dataset.dev") -Force
    Copy-Item -LiteralPath (Join-Path $goldTestFolder "$dataset.tst") `
        -Destination (Join-Path $dataRoot "$dataset.tst") -Force

    Write-Host "Prepared $dataset train/dev/test files in data/inflection."
}
