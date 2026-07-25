#pragma once
// kernel_wrapper_cuda.h — GPU-launch backend for the closed-source bench
// drivers (test/cublas_bench/CB_*, test/cudnn_bench/CD_* host_cpu.cpp).
//
// The SAME host_cpu.cpp that drives the x86-lifted kernel runs unmodified
// against a LIFTED-PTX (or vanilla-cubin) GPU kernel:
//   - const_mem[0] stays a host byte array; the driver memcpys its captured
//     param blob to +SLIFTER_ARG_BASE and pokes pointer slots.
//   - launchKernel(sym, grid, block): looks the kernel up BY NAME (the macro
//     stringifies the symbol) in the module given by $SLIFTER_MODULE
//     (PTX or cubin path), then cuLaunchKernel with the raw blob via
//     CU_LAUNCH_PARAM_BUFFER_POINTER — byte-exact param ABI, no marshalling.
//     Param size = sum of cuFuncGetParamInfo entries.
//   - The drivers' buffers are ordinary host statics/mallocs: the GPU
//     dereferences them directly via HMM (cudaDevAttrPageableMemoryAccess=1
//     on H200) — no cudaMalloc/memcpy port needed. A cudaDeviceSynchronize
//     after the launch makes results visible to the host compare.
//
// Build:  nvcc -arch=sm_90a -std=c++17 -O2 -cudart=shared -x cu \
//           -include test/common/kernel_wrapper_cuda.h host_cpu.cpp -lcuda
// Run:    SLIFTER_MODULE=/path/lifted.ptx ./a.out       (lifted)
//         SLIFTER_MODULE=/path/kernel.cubin ./a.out     (vanilla oracle)
#define SLIFTER_ARG_BASE 0x210
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

alignas(64) inline uint8_t const_mem[5][4096] = {0};

// Dummy device-symbol declarations in the drivers are satisfied by the
// extern "C" decls in each host_cpu.cpp; they are never called directly.

struct Dim3 { unsigned x = 1, y = 1, z = 1; };

namespace slifter_cuda {

inline CUmodule g_mod = nullptr;

inline void die(const char* what, CUresult r) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    fprintf(stderr, "FAIL: %s: %s (%d)\n", what, s ? s : "?", (int)r);
    exit(1);
}

inline void ensure_module() {
    if (g_mod) return;
    CUresult r = cuInit(0);
    if (r) die("cuInit", r);
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (!ctx) {
        CUdevice dev;
        if ((r = cuDeviceGet(&dev, 0))) die("cuDeviceGet", r);
        if ((r = cuDevicePrimaryCtxRetain(&ctx, dev))) die("cuDevicePrimaryCtxRetain", r);
        if ((r = cuCtxSetCurrent(ctx))) die("cuCtxSetCurrent", r);
    }
    const char* path = getenv("SLIFTER_MODULE");
    if (!path) { fprintf(stderr, "FAIL: SLIFTER_MODULE unset\n"); exit(1); }
    if ((r = cuModuleLoad(&g_mod, path))) die("cuModuleLoad", r);
}

inline size_t param_bytes(CUfunction f) {
    size_t total = 0;
    for (size_t i = 0;; ++i) {
        size_t off = 0, sz = 0;
        if (cuFuncGetParamInfo(f, i, &off, &sz) != CUDA_SUCCESS) break;
        if (off + sz > total) total = off + sz;
    }
    return total;
}

inline void launch_by_name(const char* name, Dim3 grid, Dim3 block,
                           unsigned shared_bytes = 0) {
    ensure_module();
    // Drivers never pass the dynamic-shared size (the x86 backend has no
    // limit); take it from $SLIFTER_SHARED when a kernel needs >0.
    if (const char* ss = getenv("SLIFTER_SHARED"))
        shared_bytes = (unsigned)strtoul(ss, nullptr, 0);
    CUfunction f = nullptr;
    CUresult r = cuModuleGetFunction(&f, g_mod, name);
    if (r) { fprintf(stderr, "FAIL: kernel '%s' not in module\n", name); exit(1); }
    size_t psz = param_bytes(f);
    void* blob = const_mem[0] + SLIFTER_ARG_BASE;
    void* extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, blob,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &psz,
        CU_LAUNCH_PARAM_END
    };
    void** extra_p = extra;
    // SASS-only lifted kernels take NO params — they read the captured blob
    // through the module's @const_mem global (bank 0 image). Mirror the
    // host-side const_mem[0] into it instead of passing launch params.
    CUdeviceptr cm = 0; size_t cmsz = 0;
    if (psz == 0 &&
        cuModuleGetGlobal(&cm, &cmsz, g_mod, "const_mem") == CUDA_SUCCESS) {
        size_t n = cmsz < sizeof(const_mem[0]) ? cmsz : sizeof(const_mem[0]);
        if ((r = cuMemcpyHtoD(cm, const_mem[0], n))) die("const_mem HtoD", r);
        extra_p = nullptr;
    }
    if (shared_bytes > 48 * 1024)
        cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_bytes);
    // lifted kernels can carry big stack frames (see ptxswap ensure_stack_limit)
    int lsz = 0;
    cuFuncGetAttribute(&lsz, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);
    if (lsz > 1024) cuCtxSetLimit(CU_LIMIT_STACK_SIZE, (size_t)lsz + 512);
    r = cuLaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                       shared_bytes, nullptr, nullptr, extra_p);
    if (r) die("cuLaunchKernel", r);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { fprintf(stderr, "FAIL: sync: %s\n", cudaGetErrorString(e)); exit(1); }
}

inline void launch_by_name(const char* name, unsigned g, unsigned b,
                           unsigned shared_bytes = 0) {
    launch_by_name(name, Dim3{g, 1, 1}, Dim3{b, 1, 1}, shared_bytes);
}

} // namespace slifter_cuda

// Drivers call launchKernel[2D](<symbol>, grid, block[, shared]) with the
// raw (extern "C"-declared) kernel symbol — stringify it for the by-name
// lookup. Variadic so brace commas inside Dim3{..,..} survive preprocessing.
#define launchKernel(k, ...) slifter_cuda::launch_by_name(#k, __VA_ARGS__)
#define launchKernel2D(k, ...) slifter_cuda::launch_by_name(#k, __VA_ARGS__)
