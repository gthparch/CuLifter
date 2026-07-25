#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ float d_table[16] = {
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
};

extern "C" __global__ void kern_c4_array_small(const int* idx, float* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    out[tid] = d_table[idx[tid] & 0xF];
}

int main() {
    constexpr int N=256;
    int hi[N]; float ho[N];
    for(int i=0;i<N;i++) hi[i]=i%16;
    int*di; float*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_c4_array_small<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=fabsf(ho[0]-1.0f)<1e-5f && fabsf(ho[15]-16.0f)<1e-5f;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
