#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// Force local memory spill with array too large for registers
extern "C" __global__ void kern_colsolver_local(const float* in, float* out, int N) {
    float local_key[64];  // 256 bytes → guaranteed spill to local mem
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid >= N) return;

    // Fill local array with transformed input
    for (int j = 0; j < 64 && (tid*64+j) < N*64; j++)
        local_key[j] = sinf(in[tid] + (float)j * 0.1f);

    // Complex access pattern (non-sequential → can't optimize away)
    float result = 0.0f;
    for (int iter = 0; iter < 10; iter++) {
        int idx = (iter * 7 + tid) % 64;
        result += local_key[idx];
        local_key[(idx+1)%64] = result * 0.5f;
    }
    out[tid] = result;
}

int main() {
    constexpr int N=128;
    float hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(float)(i+1)*0.01f;
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_colsolver_local<<<1,128>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=!isnanf(ho[0])&&!isinff(ho[0]);
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
