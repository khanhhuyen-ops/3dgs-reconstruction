param([string]$BasePython, [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskNative = 'external/gaussian-splatting'
$taskRevision = '414b553ef1d3032ea634b7b7a1d121173f36592c'
if (-not (Test-Path -LiteralPath $taskNative)) {
    & git clone --filter=blob:none --no-checkout https://github.com/graphdeco-inria/gaussian-splatting.git $taskNative
    if ($LASTEXITCODE) { throw 'Official source clone failed' }
    & git -C $taskNative checkout $taskRevision
    if ($LASTEXITCODE) { throw 'Pinned source checkout failed' }
}
if ((& git -C $taskNative rev-parse HEAD).Trim() -ne $taskRevision) { throw 'Existing external checkout has a different revision; preserving it.' }
& git -C $taskNative submodule update --init --recursive submodules/diff-gaussian-rasterization submodules/simple-knn
if ($LASTEXITCODE) { throw 'Pinned CUDA submodule checkout failed' }
if (-not $BasePython) {
    $BasePython = (& conda run -n ns_3dgs python -c 'import sys; print(sys.executable)' | Select-Object -Last 1).Trim()
}
if (-not (Test-Path -LiteralPath $BasePython)) { throw 'Provide -BasePython for Python 3.10 with torch 2.4.1+cu121.' }
$taskBase = Split-Path -Parent $BasePython
$env:PATH = "$taskBase;$taskBase\Library\bin;$taskBase\Scripts;$env:PATH"
$taskEnv = Join-Path $taskRoot '.tools/b0_env'
$taskPython = Join-Path $taskEnv 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    & $BasePython -m venv --system-site-packages $taskEnv
    if ($LASTEXITCODE) { throw 'venv creation failed' }
}
$taskVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$taskVs = (& $taskVswhere -latest -products '*' -property installationPath).Trim()
$taskCl = Join-Path $taskVs 'VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64'
if (-not (Test-Path -LiteralPath (Join-Path $taskCl 'cl.exe'))) { throw 'CUDA12.1 requires compatible MSVC; v14.29 was not found.' }
# Import the selected developer environment in this process only; no machine settings change.
$taskVcvars = Join-Path $taskVs 'VC/Auxiliary/Build/vcvarsall.bat'
$taskVars = & cmd.exe /d /s /c "`"$taskVcvars`" x64 -vcvars_ver=14.29 >nul && set"
if ($LASTEXITCODE) { throw 'MSVC environment initialization failed' }
foreach ($taskLine in $taskVars) {
    if ($taskLine -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
}
$env:PATH = "$taskEnv\Scripts;$taskBase;$taskBase\Library\bin;$taskBase\Scripts;$taskCl;$env:PATH"
$env:CUDA_HOME = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA/v12.1'
$env:CUDA_PATH = $env:CUDA_HOME
$env:TORCH_CUDA_ARCH_LIST = '8.6'
$env:MAX_JOBS = '2'
$env:DISTUTILS_USE_SDK = '1'
$env:MSSdk = '1'
$env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Path '.tools/b0_build_logs' -Force | Out-Null
@'
import torch
assert torch.__version__ == '2.4.1+cu121'
assert torch.cuda.is_available()
x = torch.arange(16., device='cuda:0', requires_grad=True)
x.square().sum().backward()
assert x.grad.is_cuda
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), float(x.grad.sum()))
'@ | & $taskPython -
if ($LASTEXITCODE) { throw 'Actual PyTorch CUDA forward/backward failed' }
& $taskPython -m pip install 'pytest==8.3.5'
if ($LASTEXITCODE) { throw 'Project-local test dependency installation failed' }
if (-not $SkipBuild) {
    foreach ($taskExtension in @('diff-gaussian-rasterization', 'simple-knn')) {
        $taskPreviousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $taskPython -m pip install --no-build-isolation --no-deps "external/gaussian-splatting/submodules/$taskExtension" *>&1 | Tee-Object -FilePath ".tools/b0_build_logs/$taskExtension.log"
        $taskBuildExit = $LASTEXITCODE
        $ErrorActionPreference = $taskPreviousPreference
        if ($taskBuildExit) { throw "$taskExtension build failed; inspect .tools/b0_build_logs" }
    }
}
& $taskPython -m pip freeze | Out-File -LiteralPath '.tools/b0_build_logs/requirements-resolved.txt' -Encoding utf8
Write-Output "B0 Python: $taskPython"
