此工程用于了解cupti的运行机制，通过一些c++ demo观察分析其实现原理。

目前基于cuda 11.8 版本，重点关注activity与callback的实现方式。

仅做学习使用，不做任何商业用途。

使用[spdlog](https://github.com/gabime/spdlog)作为日志库。

使用方式

1. `git clone` 本工程，根目录下执行`git submodule update --init --recursive`更新子模块。

2. 编译：`bash build.sh`

3. 运行：`./build/activity_kernel` ...
