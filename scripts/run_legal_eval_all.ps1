param(
    [string]$InputPath = "data/Legal_Dataset_V1.json",
    [string]$CorpusOutput = "runtime_data/legal_corpus_chunks.json",
    [string]$EvalOutput = "runtime_data/multiturn_evaluation_legal.json",
    [string]$IndexDir = "indexes/legal",
    [int]$TopK = 10,
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

function Resolve-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-PythonExe([string]$RepoRoot, [string]$RequestedPythonExe) {
    if (-not [string]::IsNullOrWhiteSpace($RequestedPythonExe)) {
        return $RequestedPythonExe
    }

    $venvPython = Join-Path $RepoRoot "rag_env\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    return "python"
}

function Invoke-PythonModule {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string]$Module,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [Parameter(Mandatory = $true)][string]$Title
    )

    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    Write-Host "$Python -m $Module $($Args -join ' ')" -ForegroundColor DarkGray

    & $Python -m $Module @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title (exit code $LASTEXITCODE)"
    }
}

function Get-MetricValue {
    param(
        [Parameter(Mandatory = $true)]$Metrics,
        [Parameter(Mandatory = $true)][string]$Key
    )

    $prop = $Metrics.PSObject.Properties[$Key]
    if ($null -eq $prop) {
        return $null
    }
    return $prop.Value
}

$repoRoot = Resolve-RepoRoot
Set-Location $repoRoot

$python = Resolve-PythonExe -RepoRoot $repoRoot -RequestedPythonExe $PythonExe
$resolvedPython = Get-Command $python -ErrorAction Stop

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "legal_v1_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$runOutputDir = Join-Path $repoRoot ("logs\eval_runs\{0}" -f $RunName)
New-Item -ItemType Directory -Path $runOutputDir -Force | Out-Null

Write-Host "Repo root : $repoRoot" -ForegroundColor Yellow
Write-Host "Python    : $($resolvedPython.Source)" -ForegroundColor Yellow
Write-Host "Run name  : $RunName" -ForegroundColor Yellow
Write-Host "Output dir: $runOutputDir" -ForegroundColor Yellow

Invoke-PythonModule -Python $python -Module "scripts.prepare_legal_dataset" -Title "Prepare Legal Dataset" -Args @(
    "--input-path", $InputPath,
    "--corpus-output", $CorpusOutput,
    "--eval-output", $EvalOutput
)

Invoke-PythonModule -Python $python -Module "scripts.build_index" -Title "Build Legal Index" -Args @(
    "--mode", "from_json",
    "--corpus-json", $CorpusOutput,
    "--index-dir", $IndexDir
)

$modelRuns = @(
    @{
        Name = "Model 1 Baseline"
        Module = "scripts.evaluate_model_1_baseline"
        Output = (Join-Path $runOutputDir "model_1_baseline_legal_top$TopK.json")
        Args = @("--eval-path", $EvalOutput, "--index-dir", $IndexDir, "--top-k", "$TopK")
    },
    @{
        Name = "Model 2 Rewrite Dense"
        Module = "scripts.evaluate_model_2_rewrite_dense"
        Output = (Join-Path $runOutputDir "model_2_rewrite_dense_legal_top$TopK.json")
        Args = @("--eval-path", $EvalOutput, "--index-dir", $IndexDir, "--top-k", "$TopK", "--rewrite-mode", "llm", "--max-history-turns", "6")
    },
    @{
        Name = "Model 3 Hybrid"
        Module = "scripts.evaluate_model_3_hybrid"
        Output = (Join-Path $runOutputDir "model_3_hybrid_legal_top$TopK.json")
        Args = @("--eval-path", $EvalOutput, "--index-dir", $IndexDir, "--corpus-path", $CorpusOutput, "--top-k", "$TopK")
    },
    @{
        Name = "Model 4 Hybrid Rerank"
        Module = "scripts.evaluate_model_4_hybrid_rerank"
        Output = (Join-Path $runOutputDir "model_4_hybrid_rerank_legal_top$TopK.json")
        Args = @("--eval-path", $EvalOutput, "--index-dir", $IndexDir, "--corpus-path", $CorpusOutput, "--top-k", "$TopK", "--candidate-k", "$CandidateKModel4")
    },
    @{
        Name = "Model 5 Hybrid History"
        Module = "scripts.evaluate_model_5_hybrid_history"
        Output = (Join-Path $runOutputDir "model_5_hybrid_history_legal_top$TopK.json")
        Args = @("--eval-path", $EvalOutput, "--index-dir", $IndexDir, "--corpus-path", $CorpusOutput, "--top-k", "$TopK", "--history-top-k", "$HistoryTopK", "--rewrite-mode", "llm", "--history-mode", "hybrid")
    },
    @{
        Name = "Model 6 Multi Query Hybrid"
        Module = "scripts.evaluate_model_6_multi_query_hybrid"
        Output = (Join-Path $runOutputDir "model_6_multi_query_hybrid_legal_top$TopK.json")
        Args = @("--eval-path", $EvalOutput, "--index-dir", $IndexDir, "--corpus-path", $CorpusOutput, "--top-k", "$TopK", "--history-top-k", "$HistoryTopK", "--num-queries", "$NumQueries", "--per-query-k", "$PerQueryK", "--rewrite-mode", "llm", "--history-mode", "hybrid", "--fusion-mode", "rrf")
    },
    @{
        Name = "Model 7 Full Pipeline"
        Module = "scripts.evaluate_model_7_full_pipeline"
        Output = (Join-Path $runOutputDir "model_7_full_pipeline_legal_top$TopK.json")
        Args = @("--eval-path", $EvalOutput, "--index-dir", $IndexDir, "--corpus-path", $CorpusOutput, "--top-k", "$TopK", "--candidate-k", "$CandidateKModel7", "--history-top-k", "$HistoryTopK", "--num-queries", "$NumQueries", "--per-query-k", "$PerQueryK", "--rewrite-mode", "llm", "--history-mode", "hybrid", "--fusion-mode", "rrf", "--rerank-mode", "cross_encoder")
    },
    @{
        Name = "Model 8 HyDE"
        Module = "scripts.evaluate_model_8_hyde"
        Output = (Join-Path $runOutputDir "model_8_hyde_legal_top$TopK.json")
        Args = @("--eval-path", $EvalOutput, "--index-dir", $IndexDir, "--corpus-path", $CorpusOutput, "--top-k", "$TopK", "--candidate-k", "$CandidateKModel8", "--history-top-k", "$HistoryTopK", "--rewrite-mode", "llm", "--history-mode", "hybrid", "--hyde-mode", "llm", "--include-original-query", "true", "--fusion-mode", "rrf", "--rerank-mode", "cross_encoder")
    }
)

foreach ($modelRun in $modelRuns) {
    $fullArgs = @($modelRun.Args + @("--output-path", $modelRun.Output))
    Invoke-PythonModule -Python $python -Module $modelRun.Module -Title $modelRun.Name -Args $fullArgs
}

$summaryRows = @()
$fullMetricsRows = @()

foreach ($modelRun in $modelRuns) {
    $outputPath = $modelRun.Output
    if (-not (Test-Path $outputPath)) {
        throw "Missing output file: $outputPath"
    }

    $payload = Get-Content $outputPath -Raw | ConvertFrom-Json
    $metrics = $payload.metrics
    $topKKey = "hit@{0}" -f $metrics.top_k
    $recallKey = "recall@{0}" -f $metrics.top_k

    $summaryRows += [PSCustomObject]@{
        model_name = $metrics.model_name
        top_k = [int]$metrics.top_k
        samples = [int]$metrics.samples
        hit = [double](Get-MetricValue -Metrics $metrics -Key $topKKey)
        recall = [double](Get-MetricValue -Metrics $metrics -Key $recallKey)
        mrr = [double]$metrics.mrr
        avg_latency_seconds = [double]$metrics.avg_latency_seconds
        output_file = (Split-Path $outputPath -Leaf)
    }

    $metricMap = [ordered]@{
        model_name = $metrics.model_name
        output_file = (Split-Path $outputPath -Leaf)
    }
    foreach ($property in $metrics.PSObject.Properties) {
        $metricMap[$property.Name] = $property.Value
    }
    $fullMetricsRows += [PSCustomObject]$metricMap
}

$summaryCsvPath = Join-Path $runOutputDir "summary_core_metrics.csv"
$fullMetricsCsvPath = Join-Path $runOutputDir "summary_all_metrics.csv"
$summaryJsonPath = Join-Path $runOutputDir "summary_all_metrics.json"

$summaryRows |
    Sort-Object -Property mrr -Descending |
    Export-Csv -Path $summaryCsvPath -NoTypeInformation -Encoding UTF8

$allMetricColumns = New-Object System.Collections.Generic.List[string]
$null = $allMetricColumns.Add("model_name")
$null = $allMetricColumns.Add("output_file")

foreach ($row in $fullMetricsRows) {
    foreach ($property in $row.PSObject.Properties) {
        if (-not $allMetricColumns.Contains($property.Name)) {
            $null = $allMetricColumns.Add($property.Name)
        }
    }
}

$expandedMetricRows = foreach ($row in $fullMetricsRows) {
    $expanded = [ordered]@{}
    foreach ($columnName in $allMetricColumns) {
        $prop = $row.PSObject.Properties[$columnName]
        if ($null -ne $prop) {
            $expanded[$columnName] = $prop.Value
        } else {
            $expanded[$columnName] = $null
        }
    }
    [PSCustomObject]$expanded
}

$expandedMetricRows |
    Export-Csv -Path $fullMetricsCsvPath -NoTypeInformation -Encoding UTF8

$fullMetricsRows |
    ConvertTo-Json -Depth 8 |
    Set-Content -Path $summaryJsonPath -Encoding UTF8

Write-Host ""
Write-Host "===== SUMMARY TABLE (8 MODELS) =====" -ForegroundColor Green
$summaryRows |
    Sort-Object -Property mrr -Descending |
    Format-Table -AutoSize model_name, top_k, samples, hit, recall, mrr, avg_latency_seconds, output_file

Write-Host ""
Write-Host "Saved files:" -ForegroundColor Green
Write-Host "- $summaryCsvPath"
Write-Host "- $fullMetricsCsvPath"
Write-Host "- $summaryJsonPath"
