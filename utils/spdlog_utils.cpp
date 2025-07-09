#include "spdlog_utils.h"

#include "spdlog/cfg/env.h"
#include "spdlog/sinks/stdout_color_sinks.h"
void init_spdlog() {
  // 创建一个stdout color sink（如果你不需要颜色输出，可以使用其他sink）
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

  // 创建logger并注册
  auto logger =
      std::make_shared<spdlog::logger>("logger_with_thread_id", console_sink);
  spdlog::register_logger(logger);
  spdlog::set_default_logger(logger);

  spdlog::cfg::load_env_levels(); // 读取环境变量中的日志级别

  // 设置日志格式，%t 代表线程ID
  logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%P:%t] [%s@%!:%#] %^[%L]%$ %v");
}
