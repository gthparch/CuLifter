#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void B_fsetp_ftz(const float* in, float* out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N) return;
    float val = in[i];
    // NaN-unordered compare → FSETP.GTU.FTZ or FSETP.NEU.FTZ
    bool is_nan = (val != val);           // FSETP.NEU
    bool is_positive = (val > 0.0f);      // FSETP.GT
    bool is_big = !(val <= 100.0f);       // FSETP.GTU (unordered, NaN→true)
    if (is_nan) out[i] = -999.0f;
    else if (is_big) out[i] = 100.0f;
    else if (is_positive) out[i] = val;
    else out[i] = 0.0f;
}

int main() {
    constexpr int N=256;
    float hi[N], ho[N];
    for(int i=0;i<N;i++) hi[i]=(float)(i-128);
    hi[0]=NAN; hi[1]=INFINITY; hi[2]=-0.0f; hi[3]=1e-40f; // denormal
    float*di,*do_; cudaMalloc(&di,N*4); cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    B_fsetp_ftz<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb"); fwrite(ho,4,N,f); fclose(f);
    bool ok=(ho[0]==-999.0f); // NaN → -999
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di); cudaFree(do_); return ok?0:1;
}
