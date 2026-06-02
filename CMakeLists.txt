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
find_package(TBB REQUIRED)

# 子目录
add_subdirectory(src)

# 启用 CTest
enable_testing()
add_subdirectory(tests)

# ── Install ────────────────────────────────────────────────────────────────────
install(
    TARGETS mnist_train mnist_infer
    EXPORT neuralnet-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/neuralnet
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

install(
    EXPORT neuralnet-targets
    NAMESPACE neuralnet::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuralnet
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/neuralnet-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/neuralnet-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuralnet
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/neuralnet-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/neuralnet-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/neuralnet-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuralnet
)
