// host_cpu.cpp — driver for shfl_rowreduce (41_fmha softmax QUAD butterfly).
//
// Verifies the lifter's SHFL.BFLY ^1,^2 quad reduction against a CPU oracle
// with ROW-POSITION-VARYING inputs: a lane-mapping bug shows up as a wrong
// quad-sum at specific positions (mirroring 41_fmha residual rows {34,40}).
//
// NOTE: sm_90 fixture reads args at c[0x0][0x210] => build with
//   -DSLIFTER_ARG_BASE=0x210  (the default 0x160 is the sm_80 layout and
//   would marshal the `in` pointer to the wrong const-bank offset).
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void shfl_rowreduce();

int main() {
    // 128 threads = 4 warps = 32 quads. Within a warp the ^1,^2 butterfly sums
    // the 4 lanes {4q,4q+1,4q+2,4q+3} (clamp 0x1f => within-warp segment).
    constexpr int N        = 128;
    constexpr int blockDim = 128;
    constexpr int gridDim  = 1;

    float *in  = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));

    // ROW-POSITION-VARYING inputs. Each lane's value encodes its GLOBAL quad
    // position and its within-quad lane, so ANY wrong butterfly partner (a
    // ^1<->^2 swap, or picking up a neighbor quad's lane) changes the quad sum
    // for that position (mirroring FMHA rows {34,40} going wrong).
    //   in[tid] = quad_global*1000 + (lane_in_quad+1)
    //   => quad sum = quad_global*4000 + (1+2+3+4) = quad_global*4000 + 10.
    for (int tid = 0; tid < N; tid++) {
        int warp        = tid / 32;
        int lane        = tid % 32;
        int quad_global = warp * 8 + (lane / 4);   // 0..31 across the block
        int lane_in_q   = lane % 4;                // 0..3
        in[tid] = (float)(quad_global * 1000 + (lane_in_q + 1));
    }

    launchKernel(shfl_rowreduce, gridDim, blockDim, (const float*)in, out, N);

    // CPU oracle: EXACT same quad butterfly (^1 then ^2) over the same inputs.
    // Every lane of a quad ends with the full quad sum; check all 128 outputs.
    bool ok = true;
    int nfail = 0;
    for (int warp = 0; warp < blockDim / 32; warp++) {
        for (int q = 0; q < 8; q++) {              // 8 quads per warp
            int base = warp * 32 + q * 4;
            float v0[4], v1[4], v2[4];
            for (int l = 0; l < 4; l++) v0[l] = in[base + l];
            for (int l = 0; l < 4; l++) v1[l] = v0[l] + v0[l ^ 1];      // xor 1
            for (int l = 0; l < 4; l++) v2[l] = v1[l] + v1[l ^ 2];      // xor 2
            for (int l = 0; l < 4; l++) {
                int tid = base + l;
                float exp = v2[l];
                if (fabsf(out[tid] - exp) > 1e-3f) {
                    if (nfail < 12)
                        std::cout << "FAIL tid=" << tid << " (warp " << warp
                                  << " quad " << q << " lane_in_quad " << l
                                  << "): got " << out[tid] << " expected " << exp
                                  << "\n";
                    ok = false;
                    nfail++;
                }
            }
        }
    }
    if (!ok) std::cout << "TOTAL FAILS: " << nfail << " / " << N << "\n";
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
