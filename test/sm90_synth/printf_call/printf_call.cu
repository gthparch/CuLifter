// Microbench: device printf + assert -> SASS CALL to vprintf/__assertfail.
// Exercises the external-device-function CALL lift path (FlashAttention
// imports these; see docs/developer/runtime_abi.md).
#include <cstdio>
#include <cassert>
__global__ void printf_call(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (x[i] < 0.0f) printf("neg at %d: %f\n", i, x[i]);
        assert(x[i] == x[i]);   // NaN check -> __assertfail
        y[i] = x[i] * 2.0f;
    }
}
