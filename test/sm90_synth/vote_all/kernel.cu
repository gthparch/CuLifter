// VOTE.ALL: test __all_sync with mixed predicates.
// Some lanes have pred=true, some false. __all_sync should return false
// for warps with mixed predicates, true for warps where all are true.
// Output: N ints — each thread writes 1 if __all_sync returned true, 0 otherwise.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void kern_vote_all(
    const int* __restrict__ in,
    int* __restrict__ out,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int val = in[tid];
    int pred = (val > 0) ? 1 : 0;

    unsigned mask = 0xFFFFFFFF;
    int result = __all_sync(mask, pred);

    out[tid] = result ? 1 : 0;
}

int main() {
    constexpr int N = 256;  // 8 warps

    int h_in[N];
    int h_out[N];
    memset(h_out, 0, sizeof(h_out));

    // Warp 0-3: all threads have in[i]=1 (all positive) -> __all_sync = true
    // Warp 4-7: only lane 0 has in[i]=1, rest 0 -> __all_sync = false
    for (int i = 0; i < N; i++) {
        int warp = i / 32;
        if (warp < 4) {
            h_in[i] = 1;  // all true
        } else {
            h_in[i] = ((i % 32) == 0) ? 1 : 0;  // mixed
        }
    }

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(int));

    kern_vote_all<<<1, N>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Save reference
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(int), N, f);
    fclose(f);

    // Verify
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int warp = i / 32;
        int expected = (warp < 4) ? 1 : 0;
        if (h_out[i] != expected) {
            printf("FAIL [%d] (warp %d): got %d expected %d\n", i, warp, h_out[i], expected);
            ok = false;
            break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
}
