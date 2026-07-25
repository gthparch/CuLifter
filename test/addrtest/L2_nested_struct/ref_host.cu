#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

struct Inner { float a; float b; };
struct Outer { int id; Inner inner; float c; };

extern "C" __global__ void L2_nested_struct(const Outer* arr, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = arr[i].inner.a + arr[i].inner.b + arr[i].c;
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    Outer *h_arr = new Outer[N];
    float h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) { h_arr[i] = {i, {(float)(i%7+1)*0.1f, (float)(i%5+1)*0.2f}, (float)(i%3+1)*0.3f}; }
    Outer *d_arr; float *d_out;
    cudaMalloc(&d_arr, N * sizeof(Outer));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_arr, h_arr, N * sizeof(Outer), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));
    L2_nested_struct<<<(N+255)/256, 256>>>(d_arr, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = h_arr[i].inner.a + h_arr[i].inner.b + h_arr[i].c;
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("FAIL at %d\n", i); ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_arr); cudaFree(d_out); delete[] h_arr;
    return ok ? 0 : 1;
}
