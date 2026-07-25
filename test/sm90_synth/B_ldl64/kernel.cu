#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void B_ldl64(const double* in, double* out, int N) {
    double local_arr[8]; // forces local memory for doubles
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N) return;
    // Write doubles to local → STL.64
    for (int j = 0; j < 4; j++) local_arr[j] = in[i] * (double)(j+1);
    // Non-sequential read from local → LDL.64
    double sum = 0;
    for (int j = 3; j >= 0; j--) sum += local_arr[j];
    // More local usage to prevent register optimization
    for (int j = 4; j < 8; j++) local_arr[j] = sum + (double)j;
    for (int j = 4; j < 8; j++) sum += local_arr[j];
    out[i] = sum;
}

int main() {
    constexpr int N=128;
    double hi[N], ho[N];
    for(int i=0;i<N;i++) hi[i]=(double)(i+1)*0.01;
    double*di,*do_; cudaMalloc(&di,N*8); cudaMalloc(&do_,N*8);
    cudaMemcpy(di,hi,N*8,cudaMemcpyHostToDevice);
    B_ldl64<<<1,128>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*8,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb"); fwrite(ho,8,N,f); fclose(f);
    bool ok=!isnan(ho[0])&&ho[0]!=0;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di); cudaFree(do_); return ok?0:1;
}
