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
)

target_include_directories(mnistnn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(mnistnn_core PUBLIC
    MNISTNN_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/"
)

add_executable(mnistnn
    src/main.cpp
)

target_link_libraries(mnistnn PRIVATE mnistnn_core)

enable_testing()

add_executable(mnistnn_tests
    tests/MatrixTests.cpp
)

target_link_libraries(mnistnn_tests PRIVATE mnistnn_core)

add_test(NAME mnistnn_tests COMMAND mnistnn_tests)

if(MSVC)
    target_compile_options(mnistnn_core PRIVATE /W4)
    target_compile_options(mnistnn_tests PRIVATE /W4)
else()
    target_compile_options(mnistnn_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(mnistnn_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
