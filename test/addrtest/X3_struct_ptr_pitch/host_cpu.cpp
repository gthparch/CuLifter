#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void X3_struct_ptr_pitch();

// TensorMeta is passed by value in CUDA, so SASS sees flattened fields:
//   const float* ptr, size_t pitch, int W, int H
// followed by remaining kernel params: float* out

int main() {
    constexpr int W = 32, H = 32, total = W * H;

    float *in = new float[total];
    float *out = new float[total];
    std::memset(out, 0, total * sizeof(float));
    for (int i = 0; i < total; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    size_t pitch = (size_t)W;
    Dim3 grid{(W + 15) / 16, (H + 15) / 16};
    Dim3 block{16, 16};
    // Pass struct fields flattened: ptr, pitch, W, H, then out
    launchKernel2D(X3_struct_ptr_pitch, grid, block,
                   (const float*)in, pitch, W, H, out);

    bool ok = true;
    for (int i = 0; i < total; i++) {
        if (out[i] != in[i]) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << in[i] << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
