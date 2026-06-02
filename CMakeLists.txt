if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory.")
endif()

cmake_minimum_required(VERSION 3.25)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

# 为所有目标添加编译标志
add_compile_options(-Wall -Wextra -O3 -pipe)

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
