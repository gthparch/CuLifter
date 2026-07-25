#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__constant__ float weights[4] = {1.0f, 0.5f, 0.25f, 0.125f};
__constant__ int offsets[4] = {0, 10, 20, 30};

extern "C" __global__ void kern_c4_two(const float* in, float* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    int bin = tid & 3;
    int src = tid + offsets[bin];
    if (src < N)
        out[tid] = in[src] * weights[bin];
    else
        out[tid] = 0.0f;
}

int main() {
    constexpr int N=256;
    float hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(float)(i+1)*0.1f;
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_c4_two<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    // out[0] = in[0+offsets[0]] * weights[0] = in[0]*1.0 = 0.1
    bool ok=fabsf(ho[0]-0.1f)<1e-3f;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
