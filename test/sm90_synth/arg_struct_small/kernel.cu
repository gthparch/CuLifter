#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

struct SmallParams { float* ptr; int n; float scale; };

extern "C" __global__ void kern_arg_struct_small(SmallParams p, float* out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= p.n) return;
    out[tid] = p.ptr[tid] * p.scale;
}

int main() {
    constexpr int N = 256;
    float h[N], h_out[N];
    for (int i = 0; i < N; i++) h[i] = (float)(i + 1);
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    SmallParams p = {d_in, N, 3.0f};
    kern_arg_struct_small<<<1, 256>>>(p, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin","wb"); fwrite(h_out,sizeof(float),N,f); fclose(f);
    bool ok = fabsf(h_out[0] - 3.0f) < 1e-3f;
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(d_in); cudaFree(d_out); return ok ? 0 : 1;
}
