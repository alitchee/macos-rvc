#!/bin/sh
set -e

#检查系统是否符合要求
if [ "$(uname)" = "Darwin" ]; then
  # macOS specific env:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
elif [ "$(uname)" != "Linux" ]; then
  echo "Unsupported operating system."
  exit 1
fi

#检查aria2工具是否安装
if command -v aria2c >/dev/null 2>&1; then
    echo "aria2已安装"
else
    echo " aria2 未安装 "
    echo " 安装 aria2 "
    brew install aria2 
fi

# 检查 ffmpeg 是否已安装
if command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg 已安装."
else
    echo "ffmpeg 未安装."
    echo "安装 ffmpeg..."
    brew install ffmpeg  
fi


# 检查 Miniconda 是否安装
if ! command -v conda &> /dev/null; then
    echo "Miniconda 没有安装，请先安装 Miniconda。"
    exit 1
else
    echo "Miniconda 已安装。"
    conda_root=$(conda info --root)  # 获取 Miniconda 的安装根目录
    echo "Miniconda 安装根目录为: $conda_root"
fi

condaPath="$conda_root/etc/profile.d/conda.sh"
# 初始化 Miniconda 环境
source "$condaPath"

# 提示输入环境名称
read -p "请输入环境名称: " ENV_NAME

# 如果未输入环境名称，则提示用户输入
if [[ -z "$ENV_NAME" ]]; then
    echo "没有输入环境名称，正在退出..."
    exit 1
fi

# 检查环境是否已存在
if conda env list | grep -q "$ENV_NAME"; then
    echo "环境 '$ENV_NAME' 已经存在。"
    
    # 提供删除并重新创建环境的选项
    read -p "是否删除并重新创建环境 '$ENV_NAME'？(y/n): " delete_choice
    
    if [[ "$delete_choice" == "y" || "$delete_choice" == "Y" ]]; then
        # 删除已存在的环境
        echo "正在删除环境 '$ENV_NAME'..."
        conda env remove --name "$ENV_NAME"
        
        # 重新创建环境并安装 Python 3.10
        echo "重新创建环境 '$ENV_NAME' 并安装 Python 3.10..."
        conda create --name "$ENV_NAME" python=3.10 -y
    else
        echo "跳过删除和重新创建环境的步骤。"
    fi
else
    # 创建指定的环境并安装 Python 3.10
    echo "正在创建环境 '$ENV_NAME' 并安装 Python 3.10..."
    conda create --name "$ENV_NAME" python=3.10 -y
fi

# 激活环境
echo "激活环境 '$ENV_NAME'..."
conda activate "$ENV_NAME"
# 输出当前环境
echo "当前激活的环境是: $(conda info --envs | grep '*' | awk '{print $1}')"

#配置终端代理
export http_proxy="http://127.0.0.1:10887"
export https_proxy="http://127.0.0.1:10887"
echo "已配置终端网络代理127.0.0.1:10887，请确保代理有效"
#先降级pip
pip install --upgrade "pip<24.1"
#安装pytorch-cpu版本,version<2.6，否则代码不兼容，因此先安装。对于macos,默认安装cpu版本
pip install torch==2.1.2  torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
#安装依赖
pip install -r "./requirements.txt" 

echo "依赖包安装结束"
#下载模型
# Download models
chmod +x tools/dlmodels.sh

read -p "确认是否下载模型文件？[y/n]: " confirm

if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    echo "开始下载模型文件..."
    # 继续执行原有下载逻辑
    ./tools/dlmodels.sh
elif [[ "$confirm" == "n" || "$confirm" == "N" ]]; then
    echo " 已取消下载，退出脚本。"
    exit 0
else
    echo " 无效输入，必须是 y 或 n，脚本终止。"
    exit 1
fi