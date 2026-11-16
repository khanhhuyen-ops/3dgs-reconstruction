#!/usr/bin/env bash
# Run only inside the extracted source capsule on Kaggle. Never changes host Torch/driver.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
TASK_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK_ENV="$TASK_ROOT/.tools/b0_env"
mkdir -p "$TASK_ROOT/.tools/bootstrap"
cd "$TASK_ROOT"
if [ ! -x "$TASK_ROOT/.tools/bootstrap/bin/micromamba" ]; then
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
export MAX_JOBS=2
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 Pillow==12.3.0 PyYAML==6.0.3 plyfile==1.1.3 scipy==1.15.3 ninja==1.13.2
python -c 'import torch; assert torch.__version__=="2.4.1+cu121" and torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(0); print(f"{a}.{b}")')"
nvcc --version
$CXX --version
if [ ! -d external/gaussian-splatting ]; then
  mkdir -p external
  git clone --no-checkout https://github.com/graphdeco-inria/gaussian-splatting.git external/gaussian-splatting
  git -C external/gaussian-splatting checkout 414b553ef1d3032ea634b7b7a1d121173f36592c
fi
test "$(git -C external/gaussian-splatting rev-parse HEAD)" = 414b553ef1d3032ea634b7b7a1d121173f36592c
git -C external/gaussian-splatting submodule update --init submodules/diff-gaussian-rasterization submodules/simple-knn
git -C external/gaussian-splatting/submodules/diff-gaussian-rasterization submodule update --init third_party/glm
python -m pip install --no-build-isolation --no-deps ./external/gaussian-splatting/submodules/diff-gaussian-rasterization ./external/gaussian-splatting/submodules/simple-knn
python -m pip freeze > .tools/linux-packages.txt
python -c 'from src.gaussian_splatting.native import initialize; from src.gaussian_splatting.validation import cuda_unit_fixture; from src.gaussian_splatting.config import DEFAULTS; print(initialize()); print(cuda_unit_fixture(DEFAULTS))'
