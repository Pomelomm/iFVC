# iFVC 代码仓库
修改 `train_videos.py` 中一处计算，已上传到个人github仓库：[https://ghfast.top/https://github.com/usukforever/iFVC/tree/reproduction](https://ghfast.top/https://github.com/usukforever/iFVC/tree/reproduction)

## 1. 环境要求
本文使用 CUDA Toolkit 11.6，并按官方环境选用 Ubuntu 20.04、GCC 9、Python 3.7.13 和 PyTorch 1.12.1+cu116。

本文推荐选择 `Ubuntu 20.04 + CUDA 11.6` 的服务器镜像，减少旧版 Python、PyTorch、CUDA 扩展与系统编译器之间的额外差异。显卡驱动可以高于 CUDA 11.6，但服务器中必须提供 CUDA 11.6 的 `nvcc`。

## 2. 安装系统工具
```bash
apt-get update
apt-get install -y \
  git curl aria2 unzip ffmpeg colmap \
  build-essential gcc-9 g++-9
```

## 3. 创建 Conda 环境
将仓库的 `reproduction` 分支同步到服务器后，按实际位置设置项目、数据和输出目录：

```bash
export IFVC_WORKSPACE=/root/autodl-tmp
export IFVC_ROOT="$IFVC_WORKSPACE/iFVC"
export IFVC_DATA_ROOT="$IFVC_WORKSPACE/datasets/iFVC"
export IFVC_OUTPUT_ROOT="$IFVC_WORKSPACE/outputs/iFVC_full_VRU_gz"
cd "$IFVC_ROOT"
```

```bash
conda create -n iFVC python=3.7.13 pip=22.3.1 -y \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iFVC
```

新终端继续复现时，重新加载环境和路径：

```bash
export IFVC_WORKSPACE=/root/autodl-tmp
export IFVC_ROOT="$IFVC_WORKSPACE/iFVC"
export IFVC_DATA_ROOT="$IFVC_WORKSPACE/datasets/iFVC"
export IFVC_OUTPUT_ROOT="$IFVC_WORKSPACE/outputs/iFVC_full_VRU_gz"
cd "$IFVC_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iFVC
```

## 4. 安装 PyTorch 与 Python 依赖
### 4.1 安装 PyTorch 1.12.1+cu116
```bash
cd "$IFVC_ROOT"
mkdir -p .cache/wheels

curl -fL --retry 5 \
  -o '.cache/wheels/torch-1.12.1+cu116-cp37-cp37m-linux_x86_64.whl' \
  'https://mirrors.aliyun.com/pytorch-wheels/cu116/torch-1.12.1%2Bcu116-cp37-cp37m-linux_x86_64.whl'

curl -fL --retry 5 \
  -o '.cache/wheels/torchvision-0.13.1+cu116-cp37-cp37m-linux_x86_64.whl' \
  'https://mirrors.aliyun.com/pytorch-wheels/cu116/torchvision-0.13.1%2Bcu116-cp37-cp37m-linux_x86_64.whl'

curl -fL --retry 5 \
  -o '.cache/wheels/torchaudio-0.12.1+cu116-cp37-cp37m-linux_x86_64.whl' \
  'https://mirrors.aliyun.com/pytorch-wheels/cu116/torchaudio-0.12.1%2Bcu116-cp37-cp37m-linux_x86_64.whl'

PIP_CONFIG_FILE=/dev/null \
PIP_DISABLE_PIP_VERSION_CHECK=1 \
python -m pip install --no-cache-dir \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  '.cache/wheels/torch-1.12.1+cu116-cp37-cp37m-linux_x86_64.whl' \
  '.cache/wheels/torchvision-0.13.1+cu116-cp37-cp37m-linux_x86_64.whl' \
  '.cache/wheels/torchaudio-0.12.1+cu116-cp37-cp37m-linux_x86_64.whl'
```

### 4.2 安装其余依赖
```bash
PIP_CONFIG_FILE=/dev/null \
PIP_DISABLE_PIP_VERSION_CHECK=1 \
python -m pip install --no-cache-dir \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  -r requirements-pypi-lock.txt

PIP_CONFIG_FILE=/dev/null \
PIP_DISABLE_PIP_VERSION_CHECK=1 \
python -m pip install --no-cache-dir --no-deps \
  'https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_scatter-2.1.0%2Bpt112cu116-cp37-cp37m-linux_x86_64.whl'
```

## 5. 编译项目 CUDA 扩展
`submodules/` 中的三个扩展需要在当前 Conda 环境中编译一次。重建环境、修改 PyTorch 或 CUDA 版本、更换不同计算能力的 GPU 时，需要重新编译。

本节要求系统 `nvcc` 为 CUDA 11.6（与 `torch.version.cuda` 一致）。若服务器没有 CUDA 11.6，先免安装取得 nvcc（conda 的 `cudatoolkit` 只含运行时库、不含 `nvcc`，不能用于编译）：

```bash
mkdir -p "$IFVC_WORKSPACE/.toolchain"
cd "$IFVC_WORKSPACE/.toolchain"
aria2c --continue=true --max-connection-per-server=16 --split=16 \
  --min-split-size=8M --file-allocation=none \
  --out='cuda_11.6.0_510.39.01_linux.run' \
  'https://developer.download.nvidia.cn/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run'
chmod +x cuda_11.6.0_510.39.01_linux.run
./cuda_11.6.0_510.39.01_linux.run --extract=cuda116_extract

CUDA116="$IFVC_WORKSPACE/.toolchain/cuda-11.6"
rm -rf "$CUDA116" && mkdir -p "$CUDA116/bin" "$CUDA116/nvvm" "$CUDA116/targets"
for comp in cuda116_extract/cuda_* cuda116_extract/lib*; do
  [ -d "$comp" ] || continue
  [ -d "$comp/bin" ]     && cp -a "$comp/bin/."     "$CUDA116/bin/"
  [ -d "$comp/nvvm" ]    && cp -a "$comp/nvvm/."    "$CUDA116/nvvm/"
  [ -d "$comp/targets" ] && cp -a "$comp/targets/." "$CUDA116/targets/"
done
ln -s targets/x86_64-linux/include "$CUDA116/include"
ln -s targets/x86_64-linux/lib     "$CUDA116/lib64"
ln -sf ../nvvm/bin/cicc "$CUDA116/bin/cicc"
export CUDA_HOME="$CUDA116"
```

+ 已有 CUDA 11.6 的服务器直接使用系统 `nvcc`，跳过上述块即可。

```bash
cd "$IFVC_ROOT"

unzip -q -o submodules/diff-gaussian-rasterization.zip \
  -d submodules/ -x '__MACOSX/*'
unzip -q -o submodules/simple-knn.zip \
  -d submodules/ -x '__MACOSX/*'
unzip -q -o submodules/gridencoder.zip \
  -d submodules/ -x '__MACOSX/*'

export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export MAX_JOBS=8

GPU_CC=$(python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability(0))))')
case "$GPU_CC" in
  8.0) export TORCH_CUDA_ARCH_LIST="8.0" ;;
  8.6) export TORCH_CUDA_ARCH_LIST="8.6" ;;
  8.9) export TORCH_CUDA_ARCH_LIST="8.6+PTX" ;;
esac

PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INDEX=1 \
python -m pip install --verbose --no-cache-dir \
  --no-build-isolation --no-deps ./submodules/simple-knn

PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INDEX=1 \
python -m pip install --verbose --no-cache-dir \
  --no-build-isolation --no-deps ./submodules/diff-gaussian-rasterization

PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INDEX=1 \
python -m pip install --verbose --no-cache-dir \
  --no-build-isolation --no-deps ./submodules/gridencoder
```

## 6. 下载并准备 VRU_gz 数据
### 6.1 下载并解压原始视频
```bash
mkdir -p "$IFVC_DATA_ROOT/raw/VRU_Basketball"
cd "$IFVC_DATA_ROOT/raw/VRU_Basketball"

aria2c --continue=true --max-connection-per-server=16 --split=16 \
  --min-split-size=1M --file-allocation=none \
  --out='Basketball_gz.zip' \
  'https://hf-mirror.com/datasets/BestWJH/VRU_Basketball/resolve/main/Basketball_gz.zip'

mkdir -p gz
unzip -q Basketball_gz.zip -d gz
```

### 6.2 准备训练数据和配置

`scripts/prepare_vru_gz.sh` 会从 34 路视频提取 250 帧、标定第 0 帧、为第 1–249 帧复用相机模型、准备 LPIPS 权重，并生成完整序列配置。

```bash
cd "$IFVC_ROOT"
bash scripts/prepare_vru_gz.sh all \
  --data-root "$IFVC_DATA_ROOT" \
  --output-root "$IFVC_OUTPUT_ROOT"
```

脚本使用的路径如下：

+ 原始视频：`$IFVC_DATA_ROOT/raw/VRU_Basketball/gz/videos`
+ 训练数据：`$IFVC_DATA_ROOT/prepared/VRU/Dataset/VRU_gz`
+ LPIPS 权重：`$HOME/.cache/torch/hub/checkpoints`
+ 训练配置：`$IFVC_ROOT/.cache/configs/VRU_gz_full.json`
+ 训练输出：`$IFVC_OUTPUT_ROOT`

需要单独重跑某一步时，将 `all` 换成 `extract`、`colmap`、`weights` 或 `config`，其他参数不变。

## 7. 运行完整序列复现
```bash
cd "$IFVC_ROOT"
mkdir -p "$IFVC_WORKSPACE/outputs"
set -o pipefail
python train_videos.py \
  --config_path .cache/configs/VRU_gz_full.json \
  2>&1 | tee "$IFVC_OUTPUT_ROOT.log"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "TRAIN_EXIT=$TRAIN_EXIT" | tee -a "$IFVC_OUTPUT_ROOT.log"
```



## 8. 输出目录与结果
```bash
export RESULT_DIR="$IFVC_OUTPUT_ROOT/VRU_gz/0.004_1.0"
python -m json.tool "$RESULT_DIR/frames_results.json"
```

输出目录中的主要产物如下：

+ `point_cloud/iteration_*`：模型检查点和点云；
+ `iteration_*/bitstreams`：压缩码流；
+ `test/ours_*/`：测试视角渲染图、参考图和误差图；
+ `results.json`：当前帧指标；
+ `frames_results.json`：完整序列汇总指标；
+ `outputs.log`：训练、编码、解码和评测日志。

<img src="assets/reproduction/ifvc_output_directory.png" alt="" title="null" crop="0,0,1,1" id="MPNrS" class="ne-image">

码流大小应统计 `bitstreams` 中的文件。整个 `frame*` 目录还包含模型、点云和渲染图，不能作为压缩大小。

`ours_30000` 表示使用第 30,000 次迭代保存的模型生成评测结果。PSNR 越高越好，SSIM 越接近 1 越好，LPIPS 越低越好。

