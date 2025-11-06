cmake_minimum_required(VERSION 3.20)
project(zz5 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Library
add_library(zz5 STATIC
    src/zz5.cu
    include/zz5.hpp
)

target_include_directories(zz5 PUBLIC include)

set_target_properties(zz5 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# CUDA compile flags for all GPUs
target_compile_options(zz5 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        -Xptxas=-O3,-dlcm=ca
        -maxrregcount=64
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_75,code=sm_75
        -gencode=arch=compute_80,code=sm_80
        -gencode=arch=compute_86,code=sm_86
        -gencode=arch=compute_89,code=sm_89
        -gencode=arch=compute_90,code=sm_90
        -gencode=arch=compute_90,code=compute_90
    >
)

# Example
add_executable(example examples/basic.cpp)
target_link_libraries(example PRIVATE zz5)
