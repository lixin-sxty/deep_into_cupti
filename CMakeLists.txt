# 最小 CMake 版本要求
cmake_minimum_required(VERSION 3.15)

# 定义项目名称和支持的语言
project(CUPTI_Learning_Project LANGUAGES CXX CUDA)

# 查找 CUDA 工具包
find_package(CUDA REQUIRED)

# 设置生成 compile_commands.json 文件
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_SOURCE_DIR}/compile_commands.json
    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
)

# cupti path
set(CUPTI_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include")
set(CUPTI_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64/libcupti.so")

# 设置 CUDA 编译架构
# 务必根据您的 GPU 架构修改此行，例如 "75" (Turing), "86" (Ampere) 等。
# 如果不确定，可以设置为 "native" 让 nvcc 自动检测。
set(CUDA_ARCHITECTURES "80") # <--- 务必根据您的 GPU 架构修改此行，例如 "75"

# 添加子目录，这将处理 kernel/CMakeLists.txt 并构建 cupti_kernel_lib
add_subdirectory(kernel)

# 获取demo目录下所有cu文件
file(GLOB_RECURSE DEMO_FILES demo/*.cpp)

foreach(src_file ${DEMO_FILES})
    # 获取文件名（不带路径和后缀）
    get_filename_component(src_name ${src_file} NAME_WE)

    # 添加可执行目标
    add_executable(${src_name} ${src_file})

    # 为主程序目标添加包含目录
    target_include_directories(${src_name} PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${CUPTI_INCLUDE_DIRS}
    )

    # 链接所需的库
    target_link_libraries(${src_name} PRIVATE
        ${CUDA_LIBRARIES}
        # 链接 CUPTI 库
        ${CUPTI_LIBRARIES}
        # 链接 kernel 库
        cupti_kernel_lib
    )

    # 设置主程序目标的属性
    set_target_properties(${src_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
    )
endforeach()


