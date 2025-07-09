add_library(utils STATIC spdlog_utils.cpp)

target_include_directories(utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(utils PUBLIC
    spdlog::spdlog
)

