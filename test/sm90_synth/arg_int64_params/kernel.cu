#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_arg_int64(float* out, long long stride,
                                           long long offset, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    long long idx = (long long)tid * stride + offset;
    out[tid] = (float)(idx % 1000);
}

int main() {
    constexpr int N=256;
    float ho[N]; float*do_;
    cudaMalloc(&do_,N*4);
    kern_arg_int64<<<1,256>>>(do_, 3LL, 100LL, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    // tid=0: (0*3+100)%1000 = 100
    bool ok=ho[0]==100.0f;
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %f)\n",ho[0]);
    cudaFree(do_); return ok?0:1;
}
