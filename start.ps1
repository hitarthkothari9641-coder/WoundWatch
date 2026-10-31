param([switch]$UseRetrainedSegmenter, [switch]$KaggleDemo)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot 'python'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '2'
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw 'Python was not found. Install Python, then run: python -m venv .venv; .\.venv\Scripts\python -m pip install -r service\requirements.txt'
    }
    $pythonExe = $pythonCommand.Source
    & $pythonExe -c 'import fastapi, uvicorn, torch, torchvision, PIL, numpy, pandas, sklearn, matplotlib, multipart' 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'The default Python is missing WoundWatch dependencies. Create .venv and install requirements with: python -m venv .venv; .\.venv\Scripts\python -m pip install -r service\requirements.txt'
    }
}
if ($UseRetrainedSegmenter) {
    $candidatePath = Join-Path $PSScriptRoot 'experiments\segmenter-retrain-low-memory\candidate.pt'
    if (-not (Test-Path -LiteralPath $candidatePath)) { throw 'Run the documented retraining experiment before selecting its candidate.' }
    $env:WOUNDWATCH_SEGMENTER_PATH = $candidatePath
}
$port = 8000
if ($KaggleDemo) {
    $demoRuntime = Join-Path $PSScriptRoot 'runtime\kaggle-demo-ready'
    if (-not (Test-Path -LiteralPath (Join-Path $demoRuntime 'woundwatch.sqlite'))) {
        throw 'Create the local demo first: python -m service.seed_kaggle_demo'
    }
    $env:WOUNDWATCH_DATA_DIR = $demoRuntime
    $port = 8001
}
& $pythonExe -m uvicorn service.app:app --host 127.0.0.1 --port $port --no-access-log
