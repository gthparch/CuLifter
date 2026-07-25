#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_split_scatter(const uint4* in, unsigned int* out, int N) {
    __shared__ unsigned int sdata[1024];
    int tid = threadIdx.x;
    // Load uint4 (128-bit → LDG.E.128)
    uint4 v = (tid < N) ? in[tid] : make_uint4(0,0,0,0);
    // Prefix sum of 4 elements
    unsigned int sum0 = v.x;
    unsigned int sum1 = v.y + sum0;
    unsigned int sum2 = v.z + sum1;
    unsigned int total = v.w + sum2;
    // Store intermediates to shared
    sdata[tid*4+0] = v.x; sdata[tid*4+1] = v.y;
    sdata[tid*4+2] = v.z; sdata[tid*4+3] = v.w;
    __syncthreads();
    // Write results
    if (tid < N) {
        out[tid*4+0] = 0;
        out[tid*4+1] = sum0;
        out[tid*4+2] = sum1;
        out[tid*4+3] = sum2;
    }
}

int main() {
    constexpr int N=64;
    uint4 hi[N]; unsigned int ho[N*4];
    for(int i=0;i<N;i++) hi[i]=make_uint4(1,2,3,4);
    uint4*di; unsigned int*do_;
    cudaMalloc(&di,N*16);cudaMalloc(&do_,N*4*4);
    cudaMemcpy(di,hi,N*16,cudaMemcpyHostToDevice);
    kern_split_scatter<<<1,64>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N*4,f);fclose(f);
    // prefix of {1,2,3,4}: {0,1,3,6}
    bool ok=(ho[0]==0&&ho[1]==1&&ho[2]==3&&ho[3]==6);
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
