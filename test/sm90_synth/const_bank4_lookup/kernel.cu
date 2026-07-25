#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__constant__ float lut[16] = {
    1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,
    9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f,16.0f
};

extern "C" __global__ void kern_c4_lookup(const int* idx, float* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    out[tid] = lut[idx[tid] & 0xF];
}

int main() {
    constexpr int N=256;
    int hi[N]; float ho[N];
    for(int i=0;i<N;i++) hi[i]=i%16;
    int*di; float*do_;
    cudaMalloc(&di,N*4); cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_c4_lookup<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=fabsf(ho[0]-1.0f)<1e-5f && fabsf(ho[15]-16.0f)<1e-5f;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
