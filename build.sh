#!/bin/bash

# build.sh - 自动化 CUPTI 学习项目的 CMake 编译过程

# --- 配置变量 ---
# 如果您的 CUDA Toolkit 安装在非标准路径，请在这里设置。
# 例如：CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-11.8"
# 如果不设置，CMake 会尝试自动查找。
# export CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-11.8" # <--- 确保这里是您实际的 CUDA 路径，如果需要的话取消注释并设置

# --- 编译过程 ---

# 1. 创建并进入 build 目录
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir "$BUILD_DIR"
fi
cd "$BUILD_DIR" || { echo "Failed to change to build directory. Exiting."; exit 1; }

# 2. 运行 CMake 配置
echo "Running CMake configuration..."
cmake ..

# 检查 CMake 配置是否成功
if [ $? -ne 0 ]; then
    echo "CMake configuration failed. Please check the output above for errors."
    echo "Ensure CUDA_TOOLKIT_ROOT_DIR is correctly set if CUDA is not found automatically."
    exit 1
fi

# 3. 编译项目
echo "Compiling project..."
make -j$(nproc) # 使用所有可用核心进行并行编译

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "Project compilation failed. Please check the output above for errors."
    exit 1
fi

echo "Build completed successfully!"
echo "You can now run the executable: .build/cupti_activity_demo"

# 返回到项目根目录 (可选，但通常是个好习惯)
cd ..
