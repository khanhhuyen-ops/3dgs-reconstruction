param(
    [string]$Config = 'configs/sfm_ignatius.yaml',
    [ValidateSet('preflight', 'extract', 'select', 'preprocess', 'colmap', 'all', 'custom')]
    [string]$Stage = 'preflight',
    [string]$Python = 'python',
    [switch]$DryRun,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repositoryRoot
try {
    $pipelineArguments = @('-m', 'src.pipeline', '--config', $Config, '--stage', $Stage)
    if ($DryRun) { $pipelineArguments += '--dry-run' }
    if ($Resume) { $pipelineArguments += '--resume' }
    & $Python @pipelineArguments
    $pipelineExitCode = $LASTEXITCODE
}
finally { Pop-Location }
exit $pipelineExitCode
