#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void ST_custom_struct();
int main() {
    constexpr int N = 256;
    float h_in[N+10], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N+10; i++) h_in[i] = (float)(i+1);
    // Struct Params = {scale=2.0, offset=5, bias=10.0} = 12 bytes
    float scale = 2.0f; int offset = 5; float bias = 10.0f;
    launchKernel(ST_custom_struct, (N+255)/256, 256, h_in, h_out, scale, offset, bias, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_in[i + offset] * scale + bias;
        if (fabsf(h_out[i] - exp) > 1e-3f) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
