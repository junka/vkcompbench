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
include(check_float64)
check_float64(HAVE_FLOAT64)
if(HAVE_FLOAT64)
    add_definitions(-DHAVE_FLOAT64)
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

set(COMMON_CASES
    int64
    fp64
    int32
    fp32
    int16
    fp16
    int8
)

set(DOT8_CASES
    int8dot
    int8dotaccsat
    int8dot4x8packed
)

function(add_shader SHADER_COMMON_FILE SHADER_NAME)
    set(SHADER_VECTYPE_MAP
        int64 u64vec4
        int32 uvec4
        fp64 f64vec4
        fp32 vec4
        int16 u16vec4
        fp16 f16vec4
        int8 u8vec4
        int8dot u8vec4
        int8dotaccsat u8vec4
        int8dot4x8packed uint
    )
    set(SHADER_FMA_MAP
        int64    "a*c+b"
        int32    "a*c+b"
        fp64     "a*c+b"
        fp32     "a*c+b"
        int16    "a*c+b"
        fp16     "a*c+b"
        int8     "a*c+b"
        int8dot "dotEXT(a,b)+c"
        int8dotaccsat "dotAccSatEXT(a,b,c)"
        int8dot4x8packed "dotPacked4x8EXT(a,b)+c"
    )
    list(FIND SHADER_VECTYPE_MAP ${SHADER_NAME} SHADER_INDEX)
    math(EXPR SHADER_INDEX "${SHADER_INDEX} + 1")
    list(GET SHADER_VECTYPE_MAP ${SHADER_INDEX} SHADER_VECTYPE)
    
    list(FIND SHADER_FMA_MAP ${SHADER_NAME} SHADER_INDEX)
    math(EXPR SHADER_INDEX "${SHADER_INDEX} + 1")
    list(GET SHADER_FMA_MAP ${SHADER_INDEX} SHADER_FMA)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/benchmark_${SHADER_NAME}.spv
        COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_COMMON_FILE} -c -DFMA="${SHADER_FMA}" -DVEC_TYPE="${SHADER_VECTYPE}" -o ${CMAKE_CURRENT_BINARY_DIR}/benchmark_${SHADER_NAME}.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_COMMON_FILE}
        COMMENT "Compiling ${SHADER_COMMON_FILE} to SPIR-V with VEC_TYPE=${SHADER_VECTYPE} and FMA=${SHADER_FMA}"
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/benchmark_${SHADER_NAME}.h
        COMMAND ${CMAKE_COMMAND} -D INPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/benchmark_${SHADER_NAME}.spv -D OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/benchmark_${SHADER_NAME}.h -D ARRAY_NAME=shader${SHADER_NAME} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_header.cmake
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/benchmark_${SHADER_NAME}.spv
        COMMENT "Generating benchmark_${SHADER_NAME}.h from benchmark_${SHADER_NAME}.spv"
    )
    
    list(APPEND SPIRV_FILES ${CMAKE_CURRENT_BINARY_DIR}/benchmark_${SHADER_NAME}.spv)
    list(APPEND HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/benchmark_${SHADER_NAME}.h)
    set(SPIRV_FILES ${SPIRV_FILES} PARENT_SCOPE)
    set(HEADER_FILES ${HEADER_FILES} PARENT_SCOPE)
endfunction()

foreach(SHADER_NAME ${COMMON_CASES})
    add_shader(benchmark_common.comp ${SHADER_NAME})
endforeach()

foreach(SHADER_NAME ${DOT8_CASES})
    add_shader(benchmark_int8dot_common.comp ${SHADER_NAME})
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

add_dependencies(combine_headers compile_shaders)
add_dependencies(vkcompbench generate_headers)

target_link_libraries(vkcompbench PUBLIC ${CMAKE_DL_LIBS})