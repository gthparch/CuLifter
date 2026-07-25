#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#pragma pack(push, 1)
struct Packed { char a; float b; short c; float d; };
#pragma pack(pop)

extern "C" __global__ void L3_packed_struct(const Packed* arr, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = arr[i].b + arr[i].d + (float)arr[i].c;
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    #pragma pack(push, 1)
    #pragma pack(pop)
    Packed *h_arr = new Packed[N];
    float h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) { h_arr[i] = {(char)(i%26+'a'), (float)(i%7+1)*0.1f, (short)(i%10), (float)(i%3+1)*0.2f}; }
    Packed *d_arr; float *d_out;
    cudaMalloc(&d_arr, N * sizeof(Packed));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_arr, h_arr, N * sizeof(Packed), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));
    L3_packed_struct<<<(N+255)/256, 256>>>(d_arr, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = h_arr[i].b + h_arr[i].d + (float)h_arr[i].c;
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("FAIL at %d\n", i); ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_arr); cudaFree(d_out); delete[] h_arr;
    return ok ? 0 : 1;
}
