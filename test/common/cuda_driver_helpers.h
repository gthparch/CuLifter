#pragma once

#include <cuda.h>

#include <cstdlib>
#include <iostream>

inline void checkCu(CUresult result, const char* expr, const char* file, int line) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char* err_name = nullptr;
    const char* err_string = nullptr;
    cuGetErrorName(result, &err_name);
    cuGetErrorString(result, &err_string);
    std::cerr << file << ':' << line << " CUDA driver call '" << expr << "' failed: "
              << (err_name ? err_name : "<unknown>") << " ("
              << (err_string ? err_string : "<no detail>") << ")\n";
    std::exit(EXIT_FAILURE);
}

#define CUCHK(expr) checkCu((expr), #expr, __FILE__, __LINE__)

class CuDriverSession {
public:
    CuDriverSession() {
        CUCHK(cuInit(0));
        CUCHK(cuDeviceGet(&device_, 0));
        // CUDA 13 redefined cuCtxCreate as cuCtxCreate_v4 with a
        // CUctxCreateParams* slot inserted at position 2. Pass nullptr to
        // request default behaviour; older headers that lack the field
        // are still ABI-compatible because the macro picks the right v.
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
        CUCHK(cuCtxCreate(&context_, nullptr, 0, device_));
#else
        CUCHK(cuCtxCreate(&context_, 0, device_));
#endif
    }

    ~CuDriverSession() {
        if (module_) {
            cuModuleUnload(module_);
        }
        if (context_) {
            cuCtxDestroy(context_);
        }
    }

    CUfunction loadKernel(const char* cubin_path, const char* kernel_name) {
        if (module_) {
            cuModuleUnload(module_);
            module_ = nullptr;
        }
        CUCHK(cuModuleLoad(&module_, cubin_path));
        CUfunction fn{};
        CUCHK(cuModuleGetFunction(&fn, module_, kernel_name));
        return fn;
    }

    CUcontext context() const { return context_; }

private:
    CUdevice device_{};
    CUcontext context_{};
    CUmodule module_{};
};

