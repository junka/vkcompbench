cmake_minimum_required(VERSION 3.13)

project(vkcompbench)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)
include(check_float16)
check_float16(HAVE_FLOAT16)
if(HAVE_FLOAT16)
    add_definitions(-DHAVE_FLOAT16)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wextra -Werror -fno-omit-frame-pointer")
find_package(Vulkan REQUIRED)

# Compile WGSL to SPIR-V
# add_custom_command(
#     OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/benchmark.spv
#     COMMAND naga ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.wgsl --shader-stage compute ${CMAKE_CURRENT_BINARY_DIR}/benchmark.spv
#     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.wgsl
#     COMMENT "Compiling benchmark.wgsl to SPIR-V"
# )

set(SHADER_FILES
    benchmark_int64.comp
    benchmark_fp64.comp
    benchmark_int32.comp
    benchmark_fp32.comp
    benchmark_int16.comp
    benchmark_fp16.comp
    benchmark_int8.comp
    benchmark_int8dot.comp
    benchmark_int8dotaccsat.comp
    benchmark_int8dot4x8packed.comp
)

foreach(SHADER_FILE ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    string(REGEX REPLACE "benchmark_" "" SHADER_SUFFIX ${SHADER_NAME})
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv
        COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_FILE} -c -o ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_FILE}
        COMMENT "Compiling ${SHADER_FILE} to SPIR-V"
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.h
        COMMAND ${CMAKE_COMMAND} -D INPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv -D OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.h -D ARRAY_NAME=shader${SHADER_SUFFIX} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_header.cmake
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv
        COMMENT "Generating ${SHADER_NAME}.h from ${SHADER_NAME}.spv"
    )
    
    list(APPEND SPIRV_FILES ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)
    list(APPEND HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.h)

endforeach()

add_custom_target(compile_shaders DEPENDS ${SPIRV_FILES})
add_custom_target(generate_headers DEPENDS ${HEADER_FILES})

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/benchmark.h
    COMMAND ${CMAKE_COMMAND} -D HEADER_FILES="${HEADER_FILES}" -D OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/benchmark.h -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_combined_header.cmake
    DEPENDS ${HEADER_FILES}
    COMMENT "Generating combined header benchmark.h"
)

add_custom_target(combine_headers DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/benchmark.h)
add_dependencies(generate_headers combine_headers)

include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vkcompbench main.cpp)

add_dependencies(generate_headers compile_shaders)
add_dependencies(vkcompbench generate_headers)

target_link_libraries(vkcompbench PUBLIC ${CMAKE_DL_LIBS})
