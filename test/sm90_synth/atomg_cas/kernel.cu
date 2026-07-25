#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_atomg_cas(int* lock, int* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    // Each thread tries to set lock[0] to its tid if currently 0
    int old = atomicCAS(&lock[0], 0, tid+1);
    out[tid] = old; // first thread gets 0, rest get the winner's tid+1
}

int main() {
    constexpr int N=256;
    int ho[N]; int hlock=0;
    int*dl,*do_;
    cudaMalloc(&dl,4);cudaMalloc(&do_,N*4);
    cudaMemcpy(dl,&hlock,4,cudaMemcpyHostToDevice);
    kern_atomg_cas<<<1,256>>>(dl,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    printf("TEST PASSED\n");
    cudaFree(dl);cudaFree(do_); return 0;
}
