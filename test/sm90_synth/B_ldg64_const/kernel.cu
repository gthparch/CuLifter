#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void B_ldg64_const(const double* __restrict__ in,
                                          double* __restrict__ out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N) return;
    // __ldg for constant cache path → LDG.E.64.CONSTANT
    double val = __ldg(&in[i]);
    out[i] = val * 3.0 + 1.0;
}

int main() {
    constexpr int N=256;
    double hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(double)(i+1)*0.1;
    double*di,*do_; cudaMalloc(&di,N*8);cudaMalloc(&do_,N*8);
    cudaMemcpy(di,hi,N*8,cudaMemcpyHostToDevice);
    B_ldg64_const<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*8,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb"); fwrite(ho,8,N,f); fclose(f);
    bool ok=fabs(ho[0]-(0.1*3+1))<1e-10;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
