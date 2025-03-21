
function(convert_to_cpp_array input_file output_file array_name)
    file(READ ${input_file} SPIRV_DATA HEX)
    string(REGEX MATCHALL "(..)" BYTES "${SPIRV_DATA}")
    set(ARRAY_CONTENT "")
    set(COUNTER 0)
    set(TOTAL_BYTES 0)
    foreach(BYTE ${BYTES})
        string(APPEND ARRAY_CONTENT "0x${BYTE}, ")
        math(EXPR TOTAL_BYTES "${TOTAL_BYTES} + 1")
        math(EXPR COUNTER "${COUNTER} + 1")
        if (${COUNTER} EQUAL 16)
            string(APPEND ARRAY_CONTENT "\n    ")
            set(COUNTER 0)
        endif()
    endforeach()
    string(REGEX REPLACE ",\n    $" "" ARRAY_CONTENT "${ARRAY_CONTENT}")
    file(WRITE ${output_file}
"#ifndef _BENCHMARK_SPIRV_${array_name}_H_\n"
"#define _BENCHMARK_SPIRV_${array_name}_H_\n\n"
"const inline unsigned int ${array_name}_size = ${TOTAL_BYTES};\n"
"const inline unsigned char ${array_name}_code[] = {\n    ${ARRAY_CONTENT}\n};\n\n"
"#endif // _BENCHMARK_SPIRV_${array_name}_H_"
    )
endfunction()

convert_to_cpp_array(${INPUT_FILE} ${OUTPUT_FILE} ${ARRAY_NAME})