# ML Tensor Core Workloads (SM75)

This suite contains compact ML-oriented Tensor Core kernels wired into SLifter's standard test framework.

## Workloads

- `tensorop_gemm_f16`
- `tensorop_batched_gemm_f16`
- `tensorop_gemm_bias_relu_f16`
- `tensorop_gemm_softmax_f16`

## Notes

- Device kernels use WMMA (`mma_sync`) on `sm_75`, producing `HMMA` instructions.
- Each test has both host verification paths:
  - `host_cpu.cpp` for x86 wrapper execution
  - `host_cuda.cu` for NVVM backend execution
- Generated artifacts (`*.sass`, `*.elf`, `*.ll`, `*.ptx`, `*.out`) are ignored by `.gitignore`.
