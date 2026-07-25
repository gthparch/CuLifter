#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_arg_basic(float* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    out[tid] = (float)(tid * 2);
}

int main() {
    constexpr int N=256;
    float ho[N]; float*do_;
    cudaMalloc(&do_,N*4);
    kern_arg_basic<<<1,256>>>(do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=ho[0]==0.0f && ho[1]==2.0f;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(do_); return ok?0:1;
}
