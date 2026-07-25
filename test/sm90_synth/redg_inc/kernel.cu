#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_redg_inc(unsigned int* ctr, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    atomicInc(&ctr[0], (unsigned int)(N-1));
}

int main() {
    constexpr int N=256;
    unsigned int h=0, *d;
    cudaMalloc(&d,4);cudaMemcpy(d,&h,4,cudaMemcpyHostToDevice);
    kern_redg_inc<<<1,256>>>(d,N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(&h,4,1,f);fclose(f);
    bool ok=(h==0); // 256 incs with max=255 wraps to 0
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %u)\n",h);
    cudaFree(d); return ok?0:1;
}
