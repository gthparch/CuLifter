// Unit test for the ublk_reduce_add wrapper (UBLKRED.G.S.ADD.F32 lowering).
// Verifies the flat fp32 element-wise reduce: gmem[i] += smem[i] over size_bytes,
// with the cluster-rank smem bits masked. This is the load-bearing semantic of
// the FA3 backward dQ_accum accumulation. Compile standalone.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "kernel_wrapper_warp.h"

int main() {
    const int N = 5120;                       // = UR19(0x500)*16/4 floats, FA3 bwd tile
    // shared_mem is the wrapper's global smem buffer; write the source tile there.
    float* smem = reinterpret_cast<float*>(shared_mem);
    const int SMEM_OFF = 0x10000;             // matches SASS UR18 base (+0x10000)
    float* ssrc = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(shared_mem) + SMEM_OFF);
    for (int i = 0; i < N; ++i) ssrc[i] = 0.5f * i - 3.0f;

    // gmem dst pre-seeded with a base value (dQ_accum was cleared to 0 by preprocess,
    // then accumulated across K-tiles; test a nonzero base to prove it ADDS not stores).
    static std::vector<float> gmem(N);
    for (int i = 0; i < N; ++i) gmem[i] = 100.0f + i;

    g_threadCtx.lane = 0;                      // warp-uniform single issue
    ublk_reduce_add((uint64_t)gmem.data(),
                    reinterpret_cast<uint8_t*>(shared_mem),
                    SMEM_OFF, (int64_t)N * 4);

    double maxerr = 0; int bad = 0;
    for (int i = 0; i < N; ++i) {
        float ref = (100.0f + i) + (0.5f * i - 3.0f);
        double e = std::fabs(gmem[i] - ref);
        if (e > maxerr) maxerr = e;
        if (e > 1e-4) bad++;
    }
    // Also verify lane!=0 is a no-op (warp-uniform).
    g_threadCtx.lane = 5;
    float before = gmem[0];
    ublk_reduce_add((uint64_t)gmem.data(), reinterpret_cast<uint8_t*>(shared_mem),
                    SMEM_OFF, (int64_t)N * 4);
    bool lane_noop = (gmem[0] == before);

    printf("[ublkred-unit] max_err=%.6g bad=%d/%d lane!=0 no-op=%d\n",
           maxerr, bad, N, lane_noop);
    bool pass = (maxerr < 1e-4) && (bad == 0) && lane_noop;
    printf(pass ? "TEST PASSED\n" : "TEST FAILED\n");
    return pass ? 0 : 1;
}
