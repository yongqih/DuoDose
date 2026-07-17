[CmdletBinding()]
param(
    [string]$DataDir = "data\xili_real\real_datasets",
    [string]$OutputDir = "results\final_v1",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [ValidateRange(1, 256)]
    [int]$NJobs = 4,
    [switch]$Resume,
    [switch]$Overwrite,
    [switch]$Progress,
    [switch]$NoProgress,
    [ValidateRange(0.1, 300.0)]
    [double]$ProgressRefreshSeconds = 1.0,
    [switch]$VerboseProgress,
    [ValidateRange(30, 60)]
    [int]$SubprocessHeartbeatSeconds = 45
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($Resume -and $Overwrite) {
    throw "-Resume and -Overwrite are mutually exclusive."
}
if ($Progress -and $NoProgress) { throw "-Progress and -NoProgress are mutually exclusive." }

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Protocol = Join-Path $PSScriptRoot "configs\final_protocol.yaml"
$ValidationConfig = Join-Path $PSScriptRoot "configs\final_validation_suite.yaml"
$Checker = Join-Path $PSScriptRoot "check_formal_completion.py"
$HeartbeatRunner = Join-Path $PSScriptRoot "run_with_heartbeat.py"
$StageProgressBridge = Join-Path $PSScriptRoot "progress_stage.py"
$PythonCommand = Get-Command python -ErrorAction Stop
$Python = $PythonCommand.Source
$OutputFull = if ([IO.Path]::IsPathRooted($OutputDir)) { [IO.Path]::GetFullPath($OutputDir) } else { [IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputDir)) }
$DataFull = if ([IO.Path]::IsPathRooted($DataDir)) { [IO.Path]::GetFullPath($DataDir) } else { [IO.Path]::GetFullPath((Join-Path $RepoRoot $DataDir)) }
$Logs = Join-Path $OutputFull "logs"
$RuntimeLedger = Join-Path $OutputFull "runtime_ledger.csv"
$ProgressFile = Join-Path $OutputFull "formal_progress.json"
$ProgressEnabled = -not $NoProgress
$WorkflowStart = Get-Date

if (-not (Test-Path -LiteralPath $Protocol -PathType Leaf)) { throw "Frozen protocol not found: $Protocol" }
if (-not (Test-Path -LiteralPath $ValidationConfig -PathType Leaf)) { throw "Validation config not found: $ValidationConfig" }
if (-not (Test-Path -LiteralPath $DataFull -PathType Container)) { throw "Data directory not found: $DataFull" }

New-Item -ItemType Directory -Force -Path $OutputFull, $Logs | Out-Null
$env:OMP_NUM_THREADS = [string]$NJobs
$env:MKL_NUM_THREADS = [string]$NJobs
$env:NUMEXPR_NUM_THREADS = [string]$NJobs

Push-Location $RepoRoot
try {
    $protocolJson = & $Python -c "import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))))" $Protocol
    if ($LASTEXITCODE -ne 0) { throw "Could not parse frozen protocol." }
    $Frozen = $protocolJson | ConvertFrom-Json
    $Datasets = @($Frozen.datasets.real_doublet_enriched)
    $RealApplicationDatasets = @($Frozen.datasets.real_application)
    $ControlledSeeds = @($Frozen.seeds.controlled_benchmark)
    $RealApplicationSeeds = @($Frozen.seeds.real_application)
    $ExternalMethods = @($Frozen.external_methods.methods)
    $ProtocolHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Protocol).Hash.ToLowerInvariant()
    $ValidationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ValidationConfig).Hash.ToLowerInvariant()
    $progressPayload = "$ProtocolHash|$ValidationHash|device=$Device|n_jobs=$NJobs"
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try { $ProgressConfigHash = ([BitConverter]::ToString($sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($progressPayload)))).Replace("-", "").ToLowerInvariant() } finally { $sha256.Dispose() }
    $StageDefinitions = @(
        @{ Name = "environment_and_data_validation"; Output = (Join-Path $OutputFull "data") },
        @{ Name = "controlled_benchmark"; Output = (Join-Path $OutputFull "controlled") },
        @{ Name = "real_data_application"; Output = (Join-Path $OutputFull "real_application") },
        @{ Name = "strict_domain_audit"; Output = (Join-Path $OutputFull "domain_audit") },
        @{ Name = "unified_validation_suite"; Output = (Join-Path $OutputFull "validation_suite") },
        @{ Name = "runtime_and_scalability_benchmark"; Output = (Join-Path $OutputFull "runtime") },
        @{ Name = "parameter_sensitivity"; Output = (Join-Path $OutputFull "sensitivity") },
        @{ Name = "final_aggregation"; Output = $OutputFull },
        @{ Name = "manuscript_tables"; Output = (Join-Path $OutputFull "tables") },
        @{ Name = "manuscript_figures"; Output = (Join-Path $OutputFull "figures") },
        @{ Name = "completion_check"; Output = $OutputFull }
    )
    $script:CurrentStageIndex = 0
    $script:CurrentStageStart = $null
    $script:CurrentStageStartIso = ""
    $script:CurrentStageLog = ""

    function Invoke-StageProgress([string]$Action, [string]$Status, [double]$ElapsedSeconds = 0, [int]$ExitCode = 0, [string]$FailureReason = "") {
        if (-not $ProgressEnabled) { return }
        $definition = $StageDefinitions[$script:CurrentStageIndex - 1]
        $remaining = if ($script:CurrentStageIndex -lt $StageDefinitions.Count) {
            @($StageDefinitions[$script:CurrentStageIndex..($StageDefinitions.Count - 1)] | ForEach-Object { $_.Name }) -join ","
        } else { "" }
        # PowerShell can drop empty-string arguments when invoking native
        # executables.  Passing ``--log-path ""`` therefore becomes a bare
        # ``--log-path`` token and argparse reports "expected one argument".
        # Add blankable options only when they contain a value.
        $arguments = @(
            $StageProgressBridge, "--action", $Action,
            "--ledger", $RuntimeLedger, "--snapshot", $ProgressFile,
            "--stage", [string]$definition.Name,
            "--stage-number", [string]$script:CurrentStageIndex,
            "--stage-total", [string]$StageDefinitions.Count,
            "--config-hash", $ProgressConfigHash,
            "--output-path", [string]$definition.Output,
            "--status", $Status,
            "--start-time", $script:CurrentStageStartIso,
            "--elapsed-seconds", [string]$ElapsedSeconds,
            "--exit-code", [string]$ExitCode
        )
        if (-not [string]::IsNullOrWhiteSpace([string]$script:CurrentStageLog)) {
            $arguments += @("--log-path", [string]$script:CurrentStageLog)
        }
        if (-not [string]::IsNullOrWhiteSpace($FailureReason)) {
            $arguments += @("--failure-reason", $FailureReason)
        }
        if (-not [string]::IsNullOrWhiteSpace($remaining)) {
            $arguments += @("--remaining-stages", $remaining)
        }
        & $Python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Could not update formal progress for stage $($definition.Name)." }
    }

    function Start-FormalStage([int]$Index) {
        $script:CurrentStageIndex = $Index
        $script:CurrentStageStart = Get-Date
        $script:CurrentStageStartIso = $script:CurrentStageStart.ToUniversalTime().ToString("o")
        $script:CurrentStageLog = ""
        Invoke-StageProgress "start" "RUNNING"
    }

    function Complete-FormalStage([string]$Status = "COMPLETED") {
        $elapsed = ((Get-Date) - $script:CurrentStageStart).TotalSeconds
        Invoke-StageProgress "finish" $Status $elapsed 0 ""
    }

    function Format-Command([string[]]$Arguments) {
        $display = @($Python)
        foreach ($argument in $Arguments) {
            if ($argument -match '[\s"]') { $display += ('"' + $argument.Replace('"', '\"') + '"') } else { $display += $argument }
        }
        return ($display -join ' ')
    }

    function Invoke-LoggedPython([string]$Stage, [string]$Name, [string[]]$Arguments) {
        $safeName = ($Name -replace '[^A-Za-z0-9_.-]', '_')
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $log = Join-Path $Logs ("{0}__{1}__{2}.log" -f $Stage, $safeName, $stamp)
        $childArguments = @($Arguments)
        $progressScripts = @("run_controlled_benchmark.py", "run_external_benchmark.py", "run_real_application.py", "run_domain_audit.py", "run_validation_suite.py", "run_runtime_scaling.py", "run_parameter_sensitivity.py")
        $scriptName = Split-Path -Leaf $childArguments[0]
        if ($progressScripts -contains $scriptName) {
            if ($ProgressEnabled) { $childArguments += "--progress" } else { $childArguments += "--no-progress" }
            $childArguments += @("--progress-refresh-seconds", [string]$ProgressRefreshSeconds, "--runtime-ledger", $RuntimeLedger, "--progress-file", $ProgressFile, "--progress-config-hash", $ProgressConfigHash)
            if ($VerboseProgress) { $childArguments += "--verbose-progress" }
        }
        $commandText = Format-Command $childArguments
        Write-Host ""
        Write-Host ("=== {0}: {1} ===" -f $Stage, $Name) -ForegroundColor Cyan
        Write-Host $commandText
        Set-Content -LiteralPath $log -Value ("command: " + $commandText + [Environment]::NewLine) -Encoding UTF8
        $script:CurrentStageLog = $log
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python $HeartbeatRunner --log-file $log --label ("$Stage | $Name") --heartbeat-seconds ([string]$SubprocessHeartbeatSeconds) -- $Python @childArguments
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousPreference
        if ($exitCode -ne 0) { throw "Stage '$Stage' run '$Name' failed with exit code $exitCode. See $log" }
    }

    function Clear-FormalDirectory([string]$RelativePath) {
        $target = [IO.Path]::GetFullPath((Join-Path $OutputFull $RelativePath))
        $prefix = $OutputFull.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Refusing to remove path outside output directory: $target" }
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    }

    function Get-StageStatus([string]$Stage) {
        $result = & $Python $Checker --results-dir $OutputFull --protocol $Protocol --stage $Stage --status-only --no-write 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $result) { return "INCOMPLETE" }
        return [string]($result | Select-Object -Last 1)
    }

    function Assert-StageComplete([string]$Stage) {
        $status = Get-StageStatus $Stage
        if ($status -ne "COMPLETE") { throw "Stage '$Stage' finished its command but formal output status is $status. Run the completion checker for details." }
    }

    function Test-MethodRun([string]$RunDir, [string]$MetricFile, [string[]]$Methods, [string]$Dataset, [int]$Seed, [string]$Workflow) {
        $requiredFiles = @($MetricFile, "run_manifest.json", "output_manifest.json")
        if ($MetricFile -eq "external_controlled_metrics.csv") { $requiredFiles += "external_high_RNA_operating_points.csv" }
        foreach ($name in $requiredFiles) {
            $path = Join-Path $RunDir $name
            if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) { return $false }
        }
        try {
            $metrics = @(Import-Csv -LiteralPath (Join-Path $RunDir $MetricFile))
            $manifest = Get-Content -Raw -LiteralPath (Join-Path $RunDir "run_manifest.json") | ConvertFrom-Json
        } catch { return $false }
        if ($metrics.Count -ne $Methods.Count) { return $false }
        $actual = @($metrics.method | Sort-Object -Unique)
        if (@(Compare-Object ($Methods | Sort-Object) $actual).Count -ne 0) { return $false }
        if (@($metrics | Where-Object { $_.dataset -ne $Dataset -or [int]$_.seed -ne $Seed -or $_.status.ToLowerInvariant() -ne "success" }).Count -ne 0) { return $false }
        foreach ($column in @("high_RNA_singlet_FPR", "high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall", "high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget")) {
            if ($metrics[0].PSObject.Properties.Name -notcontains $column) { return $false }
        }
        foreach ($property in @("workflow", "dataset", "seed", "protocol_config_sha256")) {
            if ($manifest.PSObject.Properties.Name -notcontains $property) { return $false }
        }
        return ($manifest.workflow -eq $Workflow -and $manifest.dataset -eq $Dataset -and [int]$manifest.seed -eq $Seed -and $manifest.protocol_config_sha256.ToLowerInvariant() -eq $ProtocolHash)
    }

    if ($Overwrite) {
        foreach ($directory in @("data", "controlled", "external", "real_application", "domain_audit", "domain_audit_inputs", "validation_suite", "runtime", "sensitivity", "tables", "figures", "reports")) {
            Clear-FormalDirectory $directory
        }
    }

    Start-FormalStage 1
    Write-Host "=== Preflight ===" -ForegroundColor Cyan
    & $Python -c "import anndata,numpy,pandas,scipy,sklearn,scrublet,torch,umap,yaml; print('Python dependencies: OK'); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
    if ($LASTEXITCODE -ne 0) { throw "Python dependency preflight failed." }
    if ($Device -eq "cuda") {
        & $Python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
        if ($LASTEXITCODE -ne 0) { throw "-Device cuda was requested but PyTorch CUDA is unavailable." }
    }
    $Rscript = & $Python -c "from duodose.r_runtime import find_rscript; p=find_rscript(); print(p if p else '')"
    if ($LASTEXITCODE -ne 0 -or -not $Rscript) { throw "Rscript was not found. Set R_SCRIPT or install R before formal external-method runs." }
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Rscript -e "packages <- c('Matrix','Seurat','SingleCellExperiment','SummarizedExperiment','scDblFinder','scds','DoubletFinder'); missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly=TRUE)]; if(length(missing)) stop(paste('Missing R packages:', paste(missing, collapse=', '))); cat('R dependencies: OK\n')"
    $rExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($rExitCode -ne 0) { throw "R dependency preflight failed." }

    $dataManifest = Join-Path $OutputFull "data\dataset_input_manifest.csv"
    $dataComplete = $false
    if ($Resume -and (Test-Path -LiteralPath $dataManifest)) {
        $prepared = @(Import-Csv -LiteralPath $dataManifest)
        $dataComplete = ($prepared.Count -eq $Datasets.Count -and @(Compare-Object ($Datasets | Sort-Object) ($prepared.dataset | Sort-Object -Unique)).Count -eq 0 -and @($prepared | Where-Object { $_.conversion_status -notin @("success", "cached") }).Count -eq 0)
    }
    if (-not $dataComplete) {
        Invoke-LoggedPython "data" "prepare_all" @("reproducibility/prepare_data.py", "--data-dir", $DataFull, "--output-dir", (Join-Path $OutputFull "data"), "--protocol", $Protocol, "--datasets", "all", "--convert-rds")
    } else { Write-Host "=== data: COMPLETE (resume) ===" -ForegroundColor Green }

    Complete-FormalStage
    Start-FormalStage 2
    $stage2Ran = $false

    if ($Resume -and (Get-StageStatus "controlled_benchmark") -eq "COMPLETE") {
        Write-Host "=== controlled benchmark: COMPLETE (resume) ===" -ForegroundColor Green
    } else {
        $stage2Ran = $true
        $arguments = @("reproducibility/run_controlled_benchmark.py", "--data-dir", $DataFull, "--datasets", "all", "--output-dir", (Join-Path $OutputFull "controlled"), "--conversion-dir", (Join-Path $OutputFull "data"), "--protocol", $Protocol, "--backends", "rf,dl", "--device", $Device, "--dl-max-epochs", [string]$Frozen.execution.dl_max_epochs, "--dl-patience", [string]$Frozen.execution.dl_patience, "--convert-rds", "--continue-on-error")
        if ([bool]$Frozen.execution.amp) { $arguments += "--amp" }
        if ($Resume) { $arguments += "--resume" }
        Invoke-LoggedPython "controlled" "all_datasets_all_seeds" $arguments
        Assert-StageComplete "controlled_benchmark"
    }

    if ($Resume) {
        # Reuse existing external scores only when they align exactly to the
        # freshly completed controlled test cells. Missing or ambiguous caches
        # remain eligible for a normal external-method rerun below.
        Invoke-LoggedPython "external" "migrate_cached_operating_points" @(
            "reproducibility/recompute_semireal_operating_points.py",
            "--controlled-dir", (Join-Path $OutputFull "controlled"),
            "--external-dir", (Join-Path $OutputFull "external"),
            "--protocol", $Protocol,
            "--external-only",
            "--skip-missing",
            "--update-manifests"
        )
    }

    foreach ($dataset in $Datasets) {
        foreach ($seedValue in $ControlledSeeds) {
            $seed = [int]$seedValue
            $runDir = Join-Path $OutputFull ("external\{0}\seed_{1}" -f $dataset, $seed)
            if ($Resume -and (Test-MethodRun $runDir "external_controlled_metrics.csv" $ExternalMethods $dataset $seed "external_benchmark")) {
                Write-Host "=== external: $dataset seed $seed COMPLETE (resume) ===" -ForegroundColor Green
                continue
            }
            $stage2Ran = $true
            Invoke-LoggedPython "external" ("{0}_seed_{1}" -f $dataset, $seed) @("reproducibility/run_external_benchmark.py", "--data-dir", $DataFull, "--dataset", $dataset, "--seed", [string]$seed, "--output-dir", $runDir, "--conversion-dir", (Join-Path $OutputFull "data"), "--protocol", $Protocol, "--methods", ($ExternalMethods -join ","), "--convert-rds")
            if (-not (Test-MethodRun $runDir "external_controlled_metrics.csv" $ExternalMethods $dataset $seed "external_benchmark")) { throw "Required external output failed validation: $dataset seed $seed" }
        }
    }
    Assert-StageComplete "external_baselines"

    if ($stage2Ran) { Complete-FormalStage } else { Complete-FormalStage "SKIPPED_VALID_CACHE" }
    Start-FormalStage 3
    $stage3Ran = $false

    if ($Resume -and (Get-StageStatus "real_data_application") -eq "COMPLETE") {
        Write-Host "=== real-data application: COMPLETE (resume) ===" -ForegroundColor Green
    } else {
        $stage3Ran = $true
        $arguments = @("reproducibility/run_real_application.py", "--data-dir", $DataFull, "--datasets", "all", "--output-dir", (Join-Path $OutputFull "real_application"), "--conversion-dir", (Join-Path $OutputFull "data"), "--protocol", $Protocol, "--external-methods", ($ExternalMethods -join ","), "--convert-rds", "--continue-on-error")
        Invoke-LoggedPython "real_application" "all_datasets" $arguments
        Assert-StageComplete "real_data_application"
    }

    if ($stage3Ran) { Complete-FormalStage } else { Complete-FormalStage "SKIPPED_VALID_CACHE" }
    Start-FormalStage 4
    $stage4Ran = $false

    if ($Resume -and (Get-StageStatus "domain_audit") -eq "COMPLETE") {
        Write-Host "=== domain audit: COMPLETE (resume) ===" -ForegroundColor Green
    } else {
        $stage4Ran = $true
        $arguments = @("reproducibility/run_domain_audit.py", "--data-dir", $DataFull, "--datasets", "all", "--output-dir", (Join-Path $OutputFull "domain_audit"), "--cache-dir", (Join-Path $OutputFull "domain_audit_inputs"), "--conversion-dir", (Join-Path $OutputFull "data"), "--protocol", $Protocol, "--max-cells-per-domain", "2000", "--convert-rds", "--continue-on-error")
        if ($Resume) { $arguments += "--resume" }
        Invoke-LoggedPython "domain_audit" "all_datasets" $arguments
        Assert-StageComplete "domain_audit"
    }

    if ($stage4Ran) { Complete-FormalStage } else { Complete-FormalStage "SKIPPED_VALID_CACHE" }
    Start-FormalStage 5
    $stage5Ran = $false

    if ($Resume -and (Get-StageStatus "validation_suite") -eq "COMPLETE") {
        Write-Host "=== validation suite: COMPLETE (resume) ===" -ForegroundColor Green
    } else {
        $stage5Ran = $true
        $arguments = @("reproducibility/run_validation_suite.py", "--config", $ValidationConfig, "--data-dir", $DataFull, "--output-dir", (Join-Path $OutputFull "validation_suite"), "--existing-domain-audit-dir", (Join-Path $OutputFull "domain_audit"), "--formal-results-dir", $OutputFull, "--mode", "full", "--dataset", [string]$Frozen.datasets.representative_dataset, "--device", $Device, "--n-jobs", [string]$NJobs)
        if ($Resume) { $arguments += "--resume" } else { $arguments += "--overwrite" }
        Invoke-LoggedPython "validation_suite" "full" $arguments
        Assert-StageComplete "validation_suite"
    }

    if ($stage5Ran) { Complete-FormalStage } else { Complete-FormalStage "SKIPPED_VALID_CACHE" }
    Start-FormalStage 6
    $stage6Ran = $false

    if ($Resume -and (Get-StageStatus "runtime_benchmark") -eq "COMPLETE") {
        Write-Host "=== runtime benchmark: COMPLETE (resume) ===" -ForegroundColor Green
    } else {
        $stage6Ran = $true
        $cellCounts = @($Frozen.runtime.cell_counts | ForEach-Object { [string]$_ }) -join ","
        $runtimeMethods = @($Frozen.runtime.methods) -join ","
        $arguments = @("reproducibility/run_runtime_scaling.py", "--data-dir", $DataFull, "--dataset", [string]$Frozen.runtime.dataset, "--output-dir", (Join-Path $OutputFull "runtime"), "--conversion-dir", (Join-Path $OutputFull "data"), "--protocol", $Protocol, "--cell-counts", $cellCounts, "--methods", $runtimeMethods, "--repetitions", [string]$Frozen.runtime.repetitions, "--device", $Device, "--n-jobs", [string]$NJobs, "--convert-rds")
        if ([bool]$Frozen.execution.amp) { $arguments += "--amp" }
        Invoke-LoggedPython "runtime" ([string]$Frozen.runtime.dataset) $arguments
        Assert-StageComplete "runtime_benchmark"
    }

    if ($stage6Ran) { Complete-FormalStage } else { Complete-FormalStage "SKIPPED_VALID_CACHE" }
    Start-FormalStage 7
    $stage7Ran = $false

    if ($Resume -and (Get-StageStatus "parameter_sensitivity") -eq "COMPLETE") {
        Write-Host "=== parameter sensitivity: COMPLETE (resume) ===" -ForegroundColor Green
    } else {
        $stage7Ran = $true
        Invoke-LoggedPython "sensitivity" ([string]$Frozen.datasets.representative_dataset) @("reproducibility/run_parameter_sensitivity.py", "--data-dir", $DataFull, "--dataset", [string]$Frozen.datasets.representative_dataset, "--output-dir", (Join-Path $OutputFull "sensitivity"), "--conversion-dir", (Join-Path $OutputFull "data"), "--protocol", $Protocol, "--convert-rds")
        Assert-StageComplete "parameter_sensitivity"
    }

    if ($stage7Ran) { Complete-FormalStage } else { Complete-FormalStage "SKIPPED_VALID_CACHE" }
    Start-FormalStage 8

    Invoke-LoggedPython "artifacts" "final_tables_figures" @("reproducibility/generate_final_artifacts.py", "--results-dir", $OutputFull, "--output-dir", $OutputFull)
    Complete-FormalStage
    Start-FormalStage 9
    Assert-StageComplete "final_tables"
    Complete-FormalStage
    Start-FormalStage 10
    Assert-StageComplete "final_figures"
    Complete-FormalStage

    Start-FormalStage 11
    Invoke-LoggedPython "completion" "final_check" @("reproducibility/check_formal_completion.py", "--results-dir", $OutputFull, "--protocol", $Protocol, "--strict")
    Complete-FormalStage
    Write-Host ""
    Write-Host "Formal analysis completion: COMPLETE" -ForegroundColor Green
    Write-Host "Output directory: $OutputFull"
    Write-Host "Completion report: $(Join-Path $OutputFull 'formal_completion_status.csv')"
    Write-Host "Per-analysis logs: $Logs"
    Write-Host "Runtime ledger: $RuntimeLedger"
    Write-Host "Live progress: $ProgressFile"
    Write-Host ("Total wall-clock time: {0:hh\:mm\:ss}" -f ((Get-Date) - $WorkflowStart))
    if (Test-Path -LiteralPath $RuntimeLedger) {
        $ledgerRows = @(Import-Csv -LiteralPath $RuntimeLedger)
        Write-Host ("Completed runs: {0}" -f @($ledgerRows | Where-Object { $_.status -in @("COMPLETED", "SUCCESS") }).Count)
        Write-Host ("Cached runs: {0}" -f @($ledgerRows | Where-Object { $_.status -in @("SKIPPED_VALID_CACHE", "CACHED") }).Count)
        Write-Host ("Failures: {0}" -f @($ledgerRows | Where-Object { $_.status -in @("FAILED", "INCOMPLETE", "INTERRUPTED") }).Count)
    }
    if (Test-Path -LiteralPath (Join-Path $OutputFull "formal_completion_status.csv")) {
        Import-Csv (Join-Path $OutputFull "formal_completion_status.csv") | Format-Table -AutoSize | Out-Host
    }
} catch {
    if ($script:CurrentStageIndex -gt 0 -and $null -ne $script:CurrentStageStart) {
        $elapsed = ((Get-Date) - $script:CurrentStageStart).TotalSeconds
        try { Invoke-StageProgress "finish" "FAILED" $elapsed 1 ([string]$_.Exception.Message) } catch { Write-Warning "Could not record failed stage progress: $_" }
    }
    throw
} finally {
    Pop-Location
}
