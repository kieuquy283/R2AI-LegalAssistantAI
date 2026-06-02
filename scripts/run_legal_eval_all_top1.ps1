param(
    [string]$InputPath = "data/Legal_Dataset_V1.json",
    [string]$CorpusOutput = "runtime_data/legal_corpus_chunks.json",
    [string]$EvalOutput = "runtime_data/multiturn_evaluation_legal.json",
    [string]$IndexDir = "indexes/legal",
    [int]$CandidateKModel4 = 30,
    [int]$CandidateKModel7 = 40,
    [int]$CandidateKModel8 = 40,
    [int]$HistoryTopK = 4,
    [int]$NumQueries = 4,
    [int]$PerQueryK = 20,
    [string]$RunName = "",
    [string]$PythonExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$baseScript = Join-Path $PSScriptRoot "run_legal_eval_all.ps1"
if (-not (Test-Path $baseScript)) {
    throw "Missing base script: $baseScript"
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "legal_v1_top1_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

& $baseScript `
    -InputPath $InputPath `
    -CorpusOutput $CorpusOutput `
    -EvalOutput $EvalOutput `
    -IndexDir $IndexDir `
    -TopK 1 `
    -CandidateKModel4 $CandidateKModel4 `
    -CandidateKModel7 $CandidateKModel7 `
    -CandidateKModel8 $CandidateKModel8 `
    -HistoryTopK $HistoryTopK `
    -NumQueries $NumQueries `
    -PerQueryK $PerQueryK `
    -RunName $RunName `
    -PythonExe $PythonExe

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
