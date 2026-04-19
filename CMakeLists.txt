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

add_executable(test_matrix tests/test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE mnistnn_core)
add_test(NAME test_matrix COMMAND test_matrix)

add_executable(test_model tests/test_model.cpp)
target_link_libraries(test_model PRIVATE mnistnn_core)
add_test(NAME test_model COMMAND test_model)

add_executable(test_train tests/test_train.cpp)
target_link_libraries(test_train PRIVATE mnistnn_core)
add_test(NAME test_train COMMAND test_train)

if(MSVC)
    target_compile_options(mnistnn_core PRIVATE /W4)
    target_compile_options(mnistnn_tests PRIVATE /W4)
else()
    target_compile_options(mnistnn_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_matrix PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_train PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_model PRIVATE -Wall -Wextra -Wpedantic)
endif()
