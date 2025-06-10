# CMakeLists.txt for CUPTI Learning Project

# 最小 CMake 版本要求
cmake_minimum_required(VERSION 3.15)

# 定义项目名称和支持的语言
# LANGUAGES CXX CUDA 表示项目使用 C++ 和 CUDA
project(CUPTI_Learning_Project LANGUAGES CXX CUDA)

# 查找 CUDA 工具包
# REQUIRED 关键字表示如果找不到 CUDA，则停止配置
find_package(CUDA REQUIRED)

# 设置 CUPTI 路径（根据你的系统调整）
# 这些变量定义是正确的
set(CUPTI_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include")
set(CUPTI_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64/libcupti.so")

# 创建 compile_commands.json 文件，用于 clangd 语言服务器
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_SOURCE_DIR}/compile_commands.json
    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
)

# 设置 CUDA 编译架构
# 对应 Makefile 中的 -arch=sm_75
# 您可以根据您的 GPU 架构进行调整，例如 "70;75;80" 或 "native"
# 务必取消注释并设置您的 GPU 架构，例如 "75" (Turing), "86" (Ampere) 等。
# 如果不确定，可以设置为 "native" 让 nvcc 自动检测。
set(CUDA_ARCHITECTURES "80") # <--- 务必取消注释并根据您的 GPU 架构修改此行

# 添加源文件 <--- 修改这里：添加 src/kernel.cu
set(SRCS
    src/main.cu
    src/kernel.cu
)

# 添加可执行文件
# cupti_demo 是最终生成的可执行文件名称
add_executable(cupti_demo ${SRCS})

# 为目标添加包含目录 <--- 关键修改：确保同时包含 CUDA 和 CUPTI 的头文件路径
target_include_directories(cupti_demo PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CUPTI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src # <--- 新增：确保可以找到 src/kernel.h
)

# 链接所需的库
# ${CUDA_LIBRARIES} 包含了 CUDA 运行时库 (cudart)
# ${CUPTI_LIBRARIES} 包含了 CUPTI 库
target_link_libraries(cupti_demo PRIVATE ${CUDA_LIBRARIES} ${CUPTI_LIBRARIES})

# 设置目标属性，确保 CUDA 架构被正确应用
set_target_properties(cupti_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
)

# (可选) 设置可执行文件的输出目录
# 默认情况下，可执行文件会生成在构建目录的根目录
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
