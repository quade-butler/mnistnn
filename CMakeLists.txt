cmake_minimum_required(VERSION 3.20)

project(mnistnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(mnistnn
    src/Dataset.cpp
    src/Matrix.cpp
    src/ModelNN.cpp
    src/TrainNN.cpp
    src/main.cpp
)

target_include_directories(mnistnn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(mnistnn PRIVATE
    MNISTNN_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/"
)

if(MSVC)
    target_compile_options(mnistnn PRIVATE /W4)
else()
    target_compile_options(mnistnn PRIVATE -Wall -Wextra -Wpedantic)
endif()
