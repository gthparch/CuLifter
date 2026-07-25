#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

struct BigParams {
    float* in_ptr; float* out_ptr;
    int width; int height; int depth;
    float alpha; float beta; int flags;
};

extern "C" __global__ void kern_arg_struct_large(BigParams p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.width * p.height;
    if (tid >= total) return;
    float val = p.in_ptr[tid];
    p.out_ptr[tid] = val * p.alpha + p.beta + (float)(p.depth * p.flags);
}

int main() {
    constexpr int W=16, H=16, N=W*H;
    float h[N], h_out[N];
    for (int i = 0; i < N; i++) h[i] = (float)(i%17+1)*0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    BigParams p = {d_in, d_out, W, H, 3, 2.0f, 1.0f, 5};
    kern_arg_struct_large<<<1, 256>>>(p);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin","wb"); fwrite(h_out,sizeof(float),N,f); fclose(f);
    float exp = h[0]*2.0f+1.0f+(float)(3*5);
    bool ok = fabsf(h_out[0]-exp)<1e-3f;
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(d_in); cudaFree(d_out); return ok?0:1;
}
