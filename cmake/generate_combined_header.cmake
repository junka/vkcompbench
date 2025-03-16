set(HEADER_FILES_LIST ${HEADER_FILES})
set(OUTPUT_FILE ${OUTPUT_FILE})

string(REPLACE " " ";" HEADER_FILES_LIST "${HEADER_FILES_LIST}")

file(WRITE ${OUTPUT_FILE} "// Combined header file\n\n")

foreach(HEADER_FILE ${HEADER_FILES_LIST})
    message(STATUS "Processing ${HEADER_FILE}")
    file(READ ${HEADER_FILE} HEADER_CONTENT)
    file(APPEND ${OUTPUT_FILE} "// Begin ${HEADER_FILE}\n")
    file(APPEND ${OUTPUT_FILE} "${HEADER_CONTENT}\n")
    file(APPEND ${OUTPUT_FILE} "// End ${HEADER_FILE}\n\n")
endforeach()

message(STATUS "Generated combined header: ${OUTPUT_FILE}")