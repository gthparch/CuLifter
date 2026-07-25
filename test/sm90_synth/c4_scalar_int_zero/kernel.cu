#include <cstdio>
#include <cuda_runtime.h>

__device__ int d_counter;  // zero-initialized, goes to c[0x4]

extern "C" __global__ void kern_c4_scalar_zero(int* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    // Read d_counter via c[4] pointer indirection
    out[tid] = d_counter + tid;
}

int main() {
    constexpr int N=256;
    int ho[N]; int*do_;
    cudaMalloc(&do_,N*4);
    kern_c4_scalar_zero<<<1,256>>>(do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    // d_counter=0, so out[i] = 0+i = i
    bool ok=(ho[0]==0 && ho[1]==1 && ho[255]==255);
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(do_); return ok?0:1;
}
