cmake_minimum_required(VERSION 3.13)

project(VulkanBenchmark)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wextra -Werror -fno-omit-frame-pointer")

find_package(Vulkan REQUIRED)

set(GLSLC glslc)
# Compile GLSL to SPIR-V
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/benchmark.spv
    COMMAND ${GLSLC} -c -O ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.comp -o ${CMAKE_CURRENT_BINARY_DIR}/benchmark.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.comp
    COMMENT "Compiling benchmark.comp to SPIR-V"
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/benchmark_spv.h
    COMMAND ${CMAKE_COMMAND} -D INPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/benchmark.spv -D OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/benchmark_spv.h -D ARRAY_NAME=shaderfp32 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_header.cmake
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/benchmark.spv
    COMMENT "Generating benchmark_spv.h from benchmark.spv"
)


include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(VulkanBenchmark main.cpp)

add_custom_target(compile_shaders DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/benchmark.spv)
add_custom_target(generate_headers DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/benchmark_spv.h)

add_dependencies(generate_headers compile_shaders)
add_dependencies(VulkanBenchmark generate_headers)


target_link_libraries(VulkanBenchmark PUBLIC ${CMAKE_DL_LIBS})