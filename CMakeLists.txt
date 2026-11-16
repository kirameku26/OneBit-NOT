cmake_minimum_required(VERSION 3.20)
project(onebit_not_ggml LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
# CPU-only: avoids Metal/BLAS overhead for one neuron.
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(ggml
  GIT_REPOSITORY https://github.com/ggml-org/ggml.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ggml)
add_executable(onebit-not-ggml ggml/main.cpp)
target_link_libraries(onebit-not-ggml PRIVATE ggml)
