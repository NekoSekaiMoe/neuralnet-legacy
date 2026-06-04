if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory.")
endif()

cmake_minimum_required(VERSION 3.25)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

# ── 构建类型 ──────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Sanitizers（仅 Debug / ASan 构建时启用）─────────────────────────────────
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# Sanitizer 与 Release 优化互斥（-O3 会产生 sanitizer false positives）— 早 fail
if((ENABLE_ASAN OR ENABLE_UBSAN) AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR
        "ENABLE_ASAN / ENABLE_UBSAN cannot be used with CMAKE_BUILD_TYPE=Release. "
        "Sanitizers work best with Debug or RelWithDebInfo (-O1 -g). "
        "Either disable sanitizers (-DENABLE_ASAN=OFF -DENABLE_UBSAN=OFF) or change "
        "the build type (-DCMAKE_BUILD_TYPE=Debug).")
endif()

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
endif()

# 为所有目标添加编译标志
add_compile_options(-Wall -Wextra -pipe)

# Release 专用优化；其他类型由 CMake 默认处理
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

# 链接 TBB（GCC libstdc++ 的 <execution> 并行策略后端）
# 需要 TBB 2021.5+ 提供现代 target 名 TBB::tbb；老版本用兼容垫片
find_package(TBB 2021.5 REQUIRED)

if(NOT TARGET TBB::tbb)
    if(TARGET tbb::tbb)
        add_library(TBB::tbb INTERFACE IMPORTED)
        target_link_libraries(TBB::tbb INTERFACE tbb::tbb)
    elseif(TARGET TBB::TBB)
        add_library(TBB::tbb INTERFACE IMPORTED)
        target_link_libraries(TBB::tbb INTERFACE TBB::TBB)
    else()
        message(FATAL_ERROR "TBB found but no recognized imported target (TBB::tbb, tbb::tbb, TBB::TBB).")
    endif()
endif()

# ── neuralnet 库（header-only INTERFACE）────────────────────────────────────
# 为下游 find_package(neuralnet) 提供可消费的目标
add_library(neuralnet INTERFACE)
add_library(neuralnet::neuralnet ALIAS neuralnet)
target_include_directories(neuralnet INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(neuralnet INTERFACE cxx_std_17)
target_link_libraries(neuralnet INTERFACE TBB::tbb)

# 子目录
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples/mnist)
endif()

# 启用 CTest
enable_testing()

# BUILD_TESTING 选项（CTest 标准约定：顶层项目默认 ON，add_subdirectory 默认 OFF）
option(BUILD_TESTING "Build tests (default ON only when this is the top-level project)" ${PROJECT_IS_TOP_LEVEL})
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# ── Install ────────────────────────────────────────────────────────────────────
# 示例程序不安装（保留 build-only 行为）
# 头文件（PATTERN 同时匹配 .h 和 .hpp，防止未来扩展漏装）
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/neuralnet
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
)

include(CMakePackageConfigHelpers)

# INTERFACE 库导出
install(TARGETS neuralnet
    EXPORT neuralnetTargets
)

install(EXPORT neuralnetTargets
    FILE neuralnetTargets.cmake
    NAMESPACE neuralnet::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuralnet
    COMPONENT dev
)

# Config & version 文件（relocatable via @PACKAGE_INIT@）
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/neuralnet-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/neuralnetConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuralnet
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/neuralnetConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/neuralnetConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/neuralnetConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuralnet
    COMPONENT dev
)
