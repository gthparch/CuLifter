#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_log_exp(const float* in, float* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    float x = in[tid];
    // exp2(log2(x)) should equal x for positive x
    out[tid] = exp2f(log2f(x));
}

int main() {
    constexpr int N=256;
    float hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(float)(i+1)*0.1f; // 0.1 to 25.6
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_log_exp<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=fabsf(ho[0]-hi[0])<0.01f;
    printf(ok?"TEST PASSED\n":"TEST FAILED (in=%f out=%f)\n",hi[0],ho[0]);
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
