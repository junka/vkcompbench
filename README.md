# vkcompbench

Do benchmark for vulkan compute pipeline

We will need ```glslc``` from shaderc projec.
- For Macos, It can be installed by: ```brew install shaderc```
- For Linux, It can be installed by: ```sudo apt install glslc```

Then compile and run:
```
cmake -S . -B build
cmake --build build
./build/vkcompbench
```




Note for ```fp64``` benchmark:
- _Float64 did not get supported in clang, use double instead

Note for ```fp16``` benchmark:
- _Float16 need gcc-12 at least

C23 features will support ```std::float16_t``` and ```std::float64_t```, but not all current active compilers will work with new features.

## testcases cover

- fp64: float point 64
- int64: integer 64
- fp32: float point 32
- int32: integer 32
- fp16: float point 16
- int16: integer 16
- int8: integer 8
- int8dot: integer 8 dot product
- int8dotaccsat: integer 8 dot product with accumulate and saturation
- int8dot4x8packed: integer 8 dot product with 4x8 packed