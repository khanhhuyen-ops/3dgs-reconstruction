param([string]$BasePython, [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$pgsrRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $pgsrRoot
$pgsrRevision = 'de24f1a38b350387e8d8fe381b2cd70c1ae946e7'
$pgsrTransforms = '89653419d0973396f3eff1a381ba09a07fffc2ed'
foreach ($pgsrSource in @(@('PGSR','https://github.com/zju3dv/PGSR.git',$pgsrRevision), @('pytorch3d','https://github.com/facebookresearch/pytorch3d.git',$pgsrTransforms))) {
    $pgsrPath = Join-Path 'external' $pgsrSource[0]
    if (-not (Test-Path -LiteralPath $pgsrPath)) {
        & git clone --filter=blob:none --no-checkout $pgsrSource[1] $pgsrPath
        if ($LASTEXITCODE) { throw 'Pinned source clone failed' }
        & git -C $pgsrPath checkout $pgsrSource[2]
        if ($LASTEXITCODE) { throw 'Pinned checkout failed' }
    }
    if ((& git -C $pgsrPath rev-parse HEAD).Trim() -ne $pgsrSource[2]) { throw 'Unexpected existing source revision; preserving checkout' }
}
if (-not $BasePython) { $BasePython = (& .\.tools\b0_env\Scripts\python.exe -c 'import sys; print(sys._base_executable)').Trim() }
if (-not (Test-Path -LiteralPath $BasePython)) { throw 'Provide a compatible Python3.10/Torch2.4.1+cu121 parent' }
$pgsrBase = Split-Path -Parent $BasePython
$env:PATH = "$pgsrBase;$pgsrBase\Library\bin;$pgsrBase\Scripts;$env:PATH"
$pgsrEnv = Join-Path $pgsrRoot '.tools/pgsr_env'
$pgsrPython = Join-Path $pgsrEnv 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pgsrPython)) {
    & $BasePython -m venv --system-site-packages $pgsrEnv
    if ($LASTEXITCODE) { throw 'PGSR venv creation failed' }
}
$pgsrVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$pgsrVs = (& $pgsrVswhere -latest -products '*' -property installationPath).Trim()
$pgsrVcvars = Join-Path $pgsrVs 'VC/Auxiliary/Build/vcvarsall.bat'
$pgsrVariables = & cmd.exe /d /s /c "`"$pgsrVcvars`" x64 -vcvars_ver=14.29 >nul && set"
if ($LASTEXITCODE) { throw 'Compatible MSVC14.29 developer environment unavailable' }
foreach ($pgsrLine in $pgsrVariables) {
    if ($pgsrLine -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1],$Matches[2],'Process') }
}
$env:PATH = "$pgsrEnv\Scripts;$pgsrBase;$pgsrBase\Library\bin;$pgsrBase\Scripts;$env:PATH"
$env:CUDA_HOME = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA/v12.1'
$env:CUDA_PATH = $env:CUDA_HOME
$env:TORCH_CUDA_ARCH_LIST = '8.6'
$env:MAX_JOBS = '2'
$env:DISTUTILS_USE_SDK = '1'
$env:MSSdk = '1'
$env:PYTHONUTF8 = '1'
$env:PYTORCH3D_NO_EXTENSION = '1'
New-Item -ItemType Directory -Path '.tools/pgsr_build_logs' -Force | Out-Null
@'
import os, sys
handles = [os.add_dll_directory(p) for p in (sys.base_prefix+'/Library/bin', sys.base_prefix+'/DLLs') if os.path.isdir(p)]
import torch
assert torch.__version__ == '2.4.1+cu121' and torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
'@ | & $pgsrPython -
if ($LASTEXITCODE) { throw 'Pinned CUDA Torch preflight failed' }
& $pgsrPython -m pip install 'opencv-python==4.10.0.84' 'fvcore==0.1.5.post20221221' 'iopath==0.1.10' 'pytest==8.3.5'
if ($LASTEXITCODE) { throw 'Project-local PGSR dependencies failed' }
# PGSR imports only pytorch3d.transforms.quaternion_to_matrix. Its official
# pure-Torch implementation executes on CUDA; unused PyTorch3D ops are not built.
& $pgsrPython -m pip install --no-build-isolation --no-deps ./external/pytorch3d
if ($LASTEXITCODE) { throw 'Pinned PyTorch3D transform installation failed' }
if (-not $SkipBuild) {
    foreach ($pgsrExtension in @('diff-plane-rasterization','simple-knn')) {
        $pgsrPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $pgsrPython -m pip install --no-build-isolation --no-deps "external/PGSR/submodules/$pgsrExtension" *>&1 | Tee-Object -FilePath ".tools/pgsr_build_logs/$pgsrExtension.log"
        $pgsrExit = $LASTEXITCODE
        $ErrorActionPreference = $pgsrPreference
        if ($pgsrExit) { throw "PGSR $pgsrExtension build failed; no CPU fallback" }
    }
}
& $pgsrPython -m pip freeze | Out-File -LiteralPath '.tools/pgsr_build_logs/requirements-resolved.txt' -Encoding utf8
Write-Output "PGSR Python: $pgsrPython"
