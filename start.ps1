param([switch]$UseRetrainedSegmenter)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot 'python'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '2'
if ($UseRetrainedSegmenter) {
    $candidatePath = Join-Path $PSScriptRoot 'experiments\segmenter-retrain-low-memory\candidate.pt'
    if (-not (Test-Path -LiteralPath $candidatePath)) { throw 'Run the documented retraining experiment before selecting its candidate.' }
    $env:WOUNDWATCH_SEGMENTER_PATH = $candidatePath
}
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -m uvicorn service.app:app --host 127.0.0.1 --port 8000 --no-access-log
