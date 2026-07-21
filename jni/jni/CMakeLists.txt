# CMakeLists.txt — builds libllama_jni.so
#
# Expects llama.cpp source to already be present at ../llama.cpp-src
# (the CI workflow clones it there before invoking CMake — see
# .github/workflows/build-llama-jni.yml). This file does not vendor
# llama.cpp itself, so the repo stays small and always builds against
# a pinned upstream commit.

cmake_minimum_required(VERSION 3.22)
project(yuvika_llama_jni CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep the binary small and CPU-only — no GPU backends, no examples/tools,
# no server, no curl dependency (model download is handled in-app already
# by LocalLLMEngine.java's downloadModel(), not by llama.cpp's downloader).
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

# ARM-specific kernels: on for arm64-v8a (all target devices have NEON/dotprod
# capable CPUs on any phone from the last several years).
if (ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp-src llama-cpp-build)

add_library(llama_jni SHARED
    llama_jni.cpp
)

target_include_directories(llama_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp-src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp-src/ggml/include
)

target_link_libraries(llama_jni PRIVATE
    llama
    log
    android
)

# Strip symbols in release to keep the .so small for phone storage.
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    target_link_options(llama_jni PRIVATE -Wl,--strip-all)
endif()
