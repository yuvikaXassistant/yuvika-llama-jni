cmake_minimum_required(VERSION 3.22)
project(llama_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to the llama.cpp checkout — passed in by the Colab build script via
# -DLLAMA_CPP_DIR=..., falls back to a sibling "llama.cpp" folder otherwise.
if(NOT DEFINED LLAMA_CPP_DIR)
    set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp)
endif()

# Keep the llama.cpp build minimal: no CLI tools/tests/server, everything
# statically linked so the final output is a single self-contained .so with
# no extra libggml.so/libllama.so to ship alongside it.
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} llama_build)

add_library(llama_jni SHARED llama_jni.cpp)

target_include_directories(llama_jni PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
)

target_link_libraries(llama_jni PRIVATE llama ggml log android)
