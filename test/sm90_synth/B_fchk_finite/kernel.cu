#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void B_fchk_finite(const float* in, float* out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N) return;
    float val = in[i];
    // isfinite → FCHK on SM90
    if (isfinite(val)) out[i] = val * 2.0f;
    else if (isnan(val)) out[i] = -1.0f;
    else out[i] = -2.0f;  // ±Inf
}

int main() {
    constexpr int N=256;
    float hi[N], ho[N];
    for(int i=0;i<N;i++) hi[i]=(float)(i+1)*0.1f;
    hi[10]=NAN; hi[20]=INFINITY; hi[30]=-INFINITY; hi[40]=0.0f;
    float*di,*do_; cudaMalloc(&di,N*4); cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    B_fchk_finite<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb"); fwrite(ho,4,N,f); fclose(f);
    bool ok=(ho[0]==0.2f && ho[10]==-1.0f && ho[20]==-2.0f && ho[30]==-2.0f && ho[40]==0.0f);
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di); cudaFree(do_); return ok?0:1;
}
