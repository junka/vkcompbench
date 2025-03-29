include(CheckCXXSourceCompiles)

macro(check_float64 HAVE_FLOAT64)
    check_cxx_source_compiles("
    int main() { 
        _Float64 f = 1.0f;
        return 0;
    }
    " HAVE_FLOAT64)

    if(HAVE_FLOAT64)
        message(STATUS "_Float64 is supported.")
    else()
        message(STATUS "_Float64 is not supported.")
    endif()

    if(HAVE_FLOAT64)
        set(HAVE_FLOAT64 TRUE CACHE INTERNAL "Support for _Float64")
    else()
        set(HAVE_FLOAT64 FALSE CACHE INTERNAL "Support for _Float64")
    endif()


endmacro(check_float64)