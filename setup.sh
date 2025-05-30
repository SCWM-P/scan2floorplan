#!/bin/bash

# 检查是否安装了conda
if ! command -v conda &> /dev/null; then
    echo "Conda未安装，请先安装Conda"
    exit 1
fi

# 创建conda环境
echo "正在创建conda环境..."
conda env create -f environment.yml

# 激活环境
echo "正在激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scan2floorplan

# 安装额外的依赖（如果需要）
echo "正在安装额外的依赖..."
pip install -r requirements.txt

# 安装jupyter内核
echo "正在安装jupyter内核..."
python -m ipykernel install --user --name=scan2floorplan

echo "环境设置完成！"
echo "使用以下命令激活环境："
echo "conda activate scan2floorplan" 