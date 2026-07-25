#include <cstdio>
#include <cuda_runtime.h>

__device__ int d_accumulator = 0;

extern "C" __global__ void kern_c4_readwrite(const int* in, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    atomicAdd(&d_accumulator, in[tid]);
}

int main() {
    constexpr int N=256;
    int hi[N];
    for(int i=0;i<N;i++) hi[i]=1;
    int*di;
    cudaMalloc(&di,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_c4_readwrite<<<1,256>>>(di,N);
    cudaDeviceSynchronize();
    int result;
    cudaMemcpyFromSymbol(&result, d_accumulator, sizeof(int));
    FILE*f=fopen("ref_output.bin","wb");fwrite(&result,4,1,f);fclose(f);
    bool ok=(result==256);
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %d, exp 256)\n",result);
    cudaFree(di); return ok?0:1;
}
