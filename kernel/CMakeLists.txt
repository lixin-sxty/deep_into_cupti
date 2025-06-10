# kernel/CMakeLists.txt

# 定义一个名为 cupti_kernel_lib 的静态库
# 这个库将由 kernel.cu 源文件编译而成
add_library(cupti_kernel_lib STATIC kernel.cu)

# 设置库的编译属性
# CUDA_SEPARABLE_COMPILATION ON 推荐用于 CUDA 库，可以提高编译速度
# CUDA_ARCHITECTURES 确保库使用与主程序相同的 GPU 架构进行编译
# 这里我们使用父目录（根 CMakeLists.txt）中定义的 CUDA_ARCHITECTURES 变量
set_target_properties(cupti_kernel_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
)

# 将当前目录（即 kernel/）添加到 cupti_kernel_lib 的公共包含路径中
# 这意味着任何链接到 cupti_kernel_lib 的目标，都可以通过 #include "kernel.h" 来找到 kernel.h
target_include_directories(cupti_kernel_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
