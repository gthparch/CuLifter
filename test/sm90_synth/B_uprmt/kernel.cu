#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void B_uprmt(const unsigned int* in, unsigned int* out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N) return;
    unsigned int val = in[i];
    // __byte_perm generates PRMT instructions
    // Swap bytes: ABCD → DCBA
    out[i] = __byte_perm(val, 0, 0x0123);
}

int main() {
    constexpr int N=256;
    unsigned int hi[N], ho[N];
    for(int i=0;i<N;i++) hi[i]=0x01020304+i;
    unsigned int*di,*do_; cudaMalloc(&di,N*4); cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    B_uprmt<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb"); fwrite(ho,4,N,f); fclose(f);
    // 0x01020304 → byte_perm with 0x0123 → 0x04030201
    bool ok=(ho[0]==0x04030201);
    printf(ok?"TEST PASSED\n":"TEST FAILED (got 0x%08x)\n",ho[0]);
    cudaFree(di); cudaFree(do_); return ok?0:1;
}
