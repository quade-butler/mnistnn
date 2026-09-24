cmake_minimum_required(VERSION 3.20)

project(mnistnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mnistnn_core
    src/dataset.cpp
    src/matrix.cpp
    src/model.cpp
    src/train.cpp
    src/eval.cpp
)

set(
    MNISTNN_MATMUL_IMPLEMENTATION
    "AUTO"
    CACHE STRING
    "Matrix multiplication implementation: NAIVE, OPTIMIZED, or AUTO"
)

set_property(
    CACHE MNISTNN_MATMUL_IMPLEMENTATION
    PROPERTY STRINGS NAIVE OPTIMIZED AUTO
)

option(
    MNISTNN_CONTROLLED_CACHE_BENCHMARK
    "Use controlled GCC flags for the matrix loop-order experiment"
    OFF
)

if(MNISTNN_MATMUL_IMPLEMENTATION STREQUAL "NAIVE")
    set(MNISTNN_MATMUL_IMPLEMENTATION_VALUE 0)
elseif(MNISTNN_MATMUL_IMPLEMENTATION STREQUAL "OPTIMIZED")
    set(MNISTNN_MATMUL_IMPLEMENTATION_VALUE 1)
elseif(MNISTNN_MATMUL_IMPLEMENTATION STREQUAL "AUTO")
    set(MNISTNN_MATMUL_IMPLEMENTATION_VALUE 2)
else()
    message(
        FATAL_ERROR
        "MNISTNN_MATMUL_IMPLEMENTATION must be NAIVE, OPTIMIZED, or AUTO"
    )
endif()

message(
    STATUS
    "Matrix multiplication implementation: ${MNISTNN_MATMUL_IMPLEMENTATION}"
)

target_include_directories(mnistnn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(mnistnn_core PUBLIC
    MNISTNN_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/"
)

target_compile_definitions(mnistnn_core PRIVATE
    MNISTNN_MATMUL_IMPLEMENTATION=${MNISTNN_MATMUL_IMPLEMENTATION_VALUE}
)

if(MNISTNN_CONTROLLED_CACHE_BENCHMARK)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(
            FATAL_ERROR
            "MNISTNN_CONTROLLED_CACHE_BENCHMARK currently requires GCC"
        )
    endif()

    target_compile_options(mnistnn_core PRIVATE
        -O2
        -fno-loop-interchange
        -fno-loop-unroll-and-jam
        -fno-tree-loop-vectorize
        -fno-tree-slp-vectorize
        -fno-unroll-loops
        -fno-peel-loops
        -fno-version-loops-for-strides
    )

    message(
        STATUS
        "Controlled cache benchmark flags enabled for mnistnn_core"
    )
endif()

add_executable(mnistnn
    src/main.cpp
)

target_link_libraries(mnistnn PRIVATE mnistnn_core)

add_executable(benchmark_matmul
    benchmarks/benchmark_matmul.cpp
)

target_link_libraries(benchmark_matmul PRIVATE mnistnn_core)

target_compile_definitions(benchmark_matmul PRIVATE
    MNISTNN_MATMUL_NAME="${MNISTNN_MATMUL_IMPLEMENTATION}"
)

enable_testing()

add_executable(test_matrix tests/test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE mnistnn_core)
add_test(NAME test_matrix COMMAND test_matrix)

add_executable(test_model tests/test_model.cpp)
target_link_libraries(test_model PRIVATE mnistnn_core)
add_test(NAME test_model COMMAND test_model)

add_executable(test_train tests/test_train.cpp)
target_link_libraries(test_train PRIVATE mnistnn_core)
add_test(NAME test_train COMMAND test_train)

add_executable(test_eval tests/test_eval.cpp)
target_link_libraries(test_eval PRIVATE mnistnn_core)
add_test(NAME test_eval COMMAND test_eval)

if(MSVC)
    target_compile_options(mnistnn_core PRIVATE /W4)
    target_compile_options(test_matrix PRIVATE /W4)
    target_compile_options(test_train PRIVATE /W4)
    target_compile_options(test_model PRIVATE /W4)
    target_compile_options(test_eval PRIVATE /W4)
    target_compile_options(benchmark_matmul PRIVATE /W4)
else()
    target_compile_options(mnistnn_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_matrix PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_train PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_model PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_eval PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(benchmark_matmul PRIVATE -Wall -Wextra -Wpedantic)
endif()
