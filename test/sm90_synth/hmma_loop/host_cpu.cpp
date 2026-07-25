#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include "kernel_wrapper.h"

extern "C" void kern_hmma_loop();

int main() {
    constexpr int N = 32;
    constexpr int iters = 4;
    uint32_t h_a[N * 4], h_b[N * 2];
    float h_out[N * 4];
    memset(h_out, 0, sizeof(h_out));
    // All inputs = packed half 1.0
    for (int i = 0; i < N * 4; i++) h_a[i] = 0x3C003C00;
    for (int i = 0; i < N * 2; i++) h_b[i] = 0x3C003C00;

    launchKernel(kern_hmma_loop, 1, 32, h_a, h_b, h_out, N, iters);

    // Compare against the GPU capture: warp-true HMMA.16816.F32 with all
    // inputs 1.0h accumulates the full k=16 dot product into EVERY fragment
    // element — d0..d3 = 16.0 per iteration = 64.0 after 4 iterations
    // (ref_output.bin is uniformly 64.0). The old hand-coded expectation
    // (iters*4.0 on d0/d2, 0 on d1/d3) encoded the obsolete lane-0 stub's
    // partial sums and went stale when HMMA became warp-true.
    std::ifstream rf("ref_output.bin", std::ios::binary);
    bool ok = true;
    if (!rf) {
        printf("FAIL: cannot open ref_output.bin\n");
        ok = false;
    } else {
        float ref[N * 4];
        rf.read((char*)ref, sizeof(ref));
        if (rf.gcount() != (std::streamsize)sizeof(ref)) {
            printf("FAIL: ref_output.bin short read\n");
            ok = false;
        } else {
            int n_fail = 0, first = -1;
            for (int i = 0; i < N * 4; i++) {
                if (std::fabs(h_out[i] - ref[i]) > 0.01f) {
                    if (first < 0) first = i;
                    n_fail++;
                }
            }
            if (n_fail) {
                printf("FAIL: %d/%d differ; first at %d got %f expected %f\n",
                       n_fail, N * 4, first, h_out[first], ref[first]);
                ok = false;
            }
        }
    }
    if (ok) printf("TEST PASSED\n");
    return ok ? 0 : 1;
}
