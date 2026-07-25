#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ int d_count = 42;
__device__ float d_scale = 2.5f;
__device__ double d_bias = 100.0;

extern "C" __global__ void kern_c4_multi(float* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    out[tid] = (float)d_count + d_scale * (float)tid + (float)d_bias;
}

int main() {
    constexpr int N=256;
    float ho[N]; float*do_;
    cudaMalloc(&do_,N*4);
    kern_c4_multi<<<1,256>>>(do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    float exp=42.0f+2.5f*0.0f+100.0f;
    bool ok=fabsf(ho[0]-exp)<0.1f;
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %f, exp %f)\n",ho[0],exp);
    cudaFree(do_); return ok?0:1;
}
