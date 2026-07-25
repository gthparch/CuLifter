#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ float d_multiplier = 1.0f;  // initial = 1.0

extern "C" __global__ void kern_c4_update(const float* in, float* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    out[tid] = in[tid] * d_multiplier;
}

int main() {
    constexpr int N=256;
    float hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(float)(i+1);
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    // Update __device__ var BEFORE launch
    float new_val = 7.0f;
    cudaMemcpyToSymbol(d_multiplier, &new_val, sizeof(float));
    kern_c4_update<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=fabsf(ho[0]-7.0f)<1e-3f;  // 1.0 * 7.0 = 7.0
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %f, exp 7.0)\n",ho[0]);
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
