# vkcompbench

Do benchmark for vulkan compute pipeline

Note for fp64 benchmark:
- _Float64 did not get supported in clang

Note for fp16 benchmark:
- _Float16 need gcc-12 at least


## testcases cover

- fp64
- int64
- fp32
- int32
- fp16
- int16
- int8
- int8dot
- int8dotaccsat
- int8dot4x8packed
