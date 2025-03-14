cmake_minimum_required(VERSION 3.13)

project(VulkanBenchmark)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wextra -Werror -fno-omit-frame-pointer")

find_package(Vulkan REQUIRED)

set(GLSLC glslc)
# Compile GLSL to SPIR-V
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/benchmark.spv
    COMMAND ${GLSLC} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.comp -o ${CMAKE_CURRENT_BINARY_DIR}/benchmark.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.comp
    COMMENT "Compiling benchmark.comp to SPIR-V"
)

include_directories(${Vulkan_INCLUDE_DIRS})

add_executable(VulkanBenchmark main.cpp)
add_custom_target(compile_shaders DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/benchmark.spv)
add_dependencies(VulkanBenchmark compile_shaders)


target_link_libraries(VulkanBenchmark PUBLIC ${CMAKE_DL_LIBS})