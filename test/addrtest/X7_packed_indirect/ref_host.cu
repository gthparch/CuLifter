#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#pragma pack(push, 1)
struct PackedEntry { char tag; float* ptr; short len; };
#pragma pack(pop)

extern "C" __global__ void X7_packed_indirect(const PackedEntry* entries,
    float* out, int N, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && i < entries[which].len)
        out[i] = entries[which].ptr[i];
}

int main(int argc, char** argv) {
    constexpr int N = 256;
    #pragma pack(push, 1)
    #pragma pack(pop)
    float *h_data = new float[N];
    float h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i%7+1)*0.1f;
    float *d_data;
    cudaMalloc(&d_data, N*sizeof(float));
    cudaMemcpy(d_data, h_data, N*sizeof(float), cudaMemcpyHostToDevice);
    PackedEntry h_entries[2] = {{'A', d_data, (short)N}, {'B', d_data, (short)N}};
    PackedEntry *d_entries;
    cudaMalloc(&d_entries, 2*sizeof(PackedEntry));
    cudaMemcpy(d_entries, h_entries, 2*sizeof(PackedEntry), cudaMemcpyHostToDevice);
    float *d_out;
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemset(d_out, 0, N*sizeof(float));
    X7_packed_indirect<<<(N+255)/256, 256>>>(d_entries, d_out, N, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_data[i]) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_data); cudaFree(d_entries); cudaFree(d_out); delete[] h_data;
    return ok ? 0 : 1;
}
