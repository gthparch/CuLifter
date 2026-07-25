#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define TW_SIZE 256
// Precomputed twiddle factors (cos/sin pairs)
__device__ float d_twiddle[TW_SIZE];

extern "C" __global__ void kern_c4_twiddle(const float* in, float* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    int tw_idx = tid % TW_SIZE;
    out[tid] = in[tid] * d_twiddle[tw_idx];
}

int main() {
    constexpr int N=256;
    // Init twiddle factors on host then copy to device
    float h_tw[TW_SIZE];
    for(int i=0;i<TW_SIZE;i++) h_tw[i]=cosf(2.0f*3.14159f*i/TW_SIZE);
    cudaMemcpyToSymbol(d_twiddle, h_tw, TW_SIZE*sizeof(float));

    float hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=1.0f;
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_c4_twiddle<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    // out[0] = 1.0 * cos(0) = 1.0
    bool ok=fabsf(ho[0]-1.0f)<1e-3f;
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %f)\n",ho[0]);
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
