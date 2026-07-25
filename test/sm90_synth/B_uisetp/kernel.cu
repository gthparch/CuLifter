#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void B_uisetp(const int* in, int* out, int N, int threshold) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N) return;
    // threshold is uniform across all threads → UISETP
    int val = in[i];
    if (val >= threshold) out[i] = val * 2;  // UISETP.GE.U32
    else if (val != 0)    out[i] = val + 1;  // UISETP.NE
    else                  out[i] = 0;
}

int main() {
    constexpr int N=256;
    int hi[N], ho[N];
    for(int i=0;i<N;i++) hi[i]=i-50;
    int*di,*do_; cudaMalloc(&di,N*4); cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    B_uisetp<<<1,256>>>(di,do_,N,100);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb"); fwrite(ho,4,N,f); fclose(f);
    printf("TEST PASSED\n");
    cudaFree(di); cudaFree(do_); return 0;
}
