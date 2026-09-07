cmake_minimum_required(VERSION 4.1)
project(hello CUDA)

set(CMAKE_CUDA_STANDARD 26)

add_executable(${PROJECT_NAME} main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(nndl)
target_link_libraries(${PROJECT_NAME} PRIVATE nndl)
