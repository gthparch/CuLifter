// VOTE.ANY: test __any_sync with a known-true predicate.
// Thread 0 always has predicate=true, so __any_sync should return true.
// Output: N ints — each thread writes 1 if __any_sync returned true, 0 otherwise.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void kern_vote_any(
    const int* __restrict__ in,
    int* __restrict__ out,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // Predicate: in[tid] > 0 (we'll set in[0]=1, others=0 per warp)
    int val = in[tid];
    int pred = (val > 0) ? 1 : 0;

    unsigned mask = 0xFFFFFFFF;
    int result = __any_sync(mask, pred);

    out[tid] = result ? 1 : 0;
}

int main() {
    constexpr int N = 256;

    int h_in[N];
    int h_out[N];
    memset(h_out, 0, sizeof(h_out));

    // Only lane 0 of each warp has a positive value
    for (int i = 0; i < N; i++) {
        h_in[i] = ((i % 32) == 0) ? 1 : 0;
    }

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(int));

    kern_vote_any<<<1, N>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Save reference
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(int), N, f);
    fclose(f);

    // Verify: __any_sync should be true for all threads (lane 0 has pred=1)
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != 1) {
            printf("FAIL [%d]: got %d expected 1\n", i, h_out[i]);
            ok = false;
            break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
}
