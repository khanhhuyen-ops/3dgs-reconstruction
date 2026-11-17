#!/usr/bin/env bash
# Independent PGSR environment. No changes to host Torch or B0 environment.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
TASK_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK_ENV="$TASK_ROOT/.tools/pgsr_env"
mkdir -p "$TASK_ROOT/.tools/bootstrap"
cd "$TASK_ROOT"
if [ ! -x .tools/bootstrap/bin/micromamba ]; then
  curl --fail --location https://micro.mamba.pm/api/micromamba/linux-64/2.3.2 -o .tools/bootstrap/micromamba.tar.bz2
  tar -xjf .tools/bootstrap/micromamba.tar.bz2 -C .tools/bootstrap bin/micromamba
  sha256sum .tools/bootstrap/micromamba.tar.bz2 > .tools/bootstrap/download.sha256
fi
export MAMBA_ROOT_PREFIX="$TASK_ROOT/.tools/mamba"
if [ ! -x "$TASK_ENV/bin/python" ]; then
  .tools/bootstrap/bin/micromamba create -y -p "$TASK_ENV" -c nvidia/label/cuda-12.1.1 -c conda-forge \
    python=3.10.20 pip cuda-toolkit=12.1.1 gcc_linux-64=12 gxx_linux-64=12 ninja
fi
export PATH="$TASK_ENV/bin:$PATH"
export CUDA_HOME="$TASK_ENV"
export CC=x86_64-conda-linux-gnu-cc
export CXX=x86_64-conda-linux-gnu-c++
export CUDAHOSTCXX="$TASK_ENV/bin/$CXX"
export MAX_JOBS=2 PIP_DISABLE_PIP_VERSION_CHECK=1
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 Pillow==12.3.0 PyYAML==6.0.3 plyfile==1.1.3 scipy==1.15.3 ninja==1.13.2 \
  opencv-python-headless==4.10.0.84 fvcore==0.1.5.post20221221 iopath==0.1.10 pytest==8.3.5 tqdm
python -c 'import torch; assert torch.__version__=="2.4.1+cu121" and torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(0); print(f"{a}.{b}")')"
mkdir -p external
if [ ! -d external/PGSR ]; then
  git clone --no-checkout https://github.com/zju3dv/PGSR.git external/PGSR
  git -C external/PGSR checkout de24f1a38b350387e8d8fe381b2cd70c1ae946e7
fi
test "$(git -C external/PGSR rev-parse HEAD)" = de24f1a38b350387e8d8fe381b2cd70c1ae946e7
if [ ! -d external/pytorch3d ]; then
  git clone --no-checkout https://github.com/facebookresearch/pytorch3d.git external/pytorch3d
  git -C external/pytorch3d checkout 89653419d0973396f3eff1a381ba09a07fffc2ed
fi
test "$(git -C external/pytorch3d rev-parse HEAD)" = 89653419d0973396f3eff1a381ba09a07fffc2ed
PYTORCH3D_NO_EXTENSION=1 python -m pip install --no-build-isolation --no-deps ./external/pytorch3d
python -m pip install --no-build-isolation --no-deps ./external/PGSR/submodules/diff-plane-rasterization ./external/PGSR/submodules/simple-knn
python -m pip freeze > .tools/pgsr-linux-packages.txt
nvcc --version
$CXX --version
python -c 'from src.pgsr.native import initialize; print(initialize())'
