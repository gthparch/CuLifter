#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void B_p2r(const int* in, int* out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N) return;
    int val = in[i];
    // Multiple predicates packed into integer bits → P2R
    bool a = (val > 0);
    bool b = (val < 100);
    bool c = (val != 50);
    bool d = (val > 25);
    out[i] = (a?1:0) | (b?2:0) | (c?4:0) | (d?8:0);
}

int main() {
    constexpr int N=256;
    int hi[N], ho[N];
    for(int i=0;i<N;i++) hi[i]=i;
    int*di,*do_; cudaMalloc(&di,N*4); cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    B_p2r<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb"); fwrite(ho,4,N,f); fclose(f);
    // i=1: a=T,b=T,c=T,d=F → 1+2+4+0=7
    bool ok=(ho[1]==7);
    printf(ok?"TEST PASSED\n":"TEST FAILED (ho[1]=%d)\n",ho[1]);
    cudaFree(di); cudaFree(do_); return ok?0:1;
}
