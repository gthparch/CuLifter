#include <cstdio>
#include <cuda_runtime.h>

__constant__ int permutation[8] = {7, 3, 5, 1, 6, 0, 4, 2};

extern "C" __global__ void kern_c4_int(const int* in, int* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    int idx = permutation[tid & 7];
    out[tid] = in[idx + (tid & ~7)]; // permute within groups of 8
}

int main() {
    constexpr int N=256;
    int hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=i+100;
    int*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_c4_int<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    // out[0] = in[permutation[0]] = in[7] = 107
    bool ok=(ho[0]==107);
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %d)\n",ho[0]);
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
