#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_mixed_types(
    float* out, int i_val, float f_val, double d_val,
    long long ll_val, short s_val, int N
) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    out[tid]=(float)i_val+(float)f_val+(float)d_val+(float)ll_val+(float)s_val+(float)tid;
}

int main() {
    constexpr int N=256;
    float ho[N]; float*do_;
    cudaMalloc(&do_,N*4);
    kern_mixed_types<<<1,256>>>(do_, 10, 2.5f, 100.0, 1000LL, (short)5, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    float exp=10+2.5f+100.0f+1000.0f+5.0f+0.0f;
    bool ok=fabsf(ho[0]-exp)<0.1f;
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %f exp %f)\n",ho[0],exp);
    cudaFree(do_); return ok?0:1;
}
