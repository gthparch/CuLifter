#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void WS5_ballot_popc(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < N) ? in[tid] : 0;
    unsigned mask = __ballot_sync(0xffffffff, val > 0);
    int count = __popc(mask);
    if (tid < N) out[tid] = count;
}

int main(int argc, char** argv) {
    constexpr int N = 256;
    int h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (i % 3 == 0) ? 1 : -1;
    memset(h_out, 0, sizeof(h_out));
    int *d_in, *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    WS5_ballot_popc<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    // Each warp: count how many lanes have val > 0
    bool ok = true;
    for (int w = 0; w < N/32; w++) {
        int cnt = 0;
        for (int i = w*32; i < (w+1)*32; i++) if (h_in[i] > 0) cnt++;
        for (int i = w*32; i < (w+1)*32; i++)
            if (h_out[i] != cnt) { printf("FAIL at %d: %d vs %d\n", i, h_out[i], cnt); ok=false; goto done; }
    } done:
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out); return ok ? 0 : 1;
}
