#include <cstdio>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__ unsigned int scanwarp_v(unsigned int val, volatile unsigned int* s, int maxlvl) {
    int lid = threadIdx.x;
    int idx = 2*lid - (lid & (WARP_SIZE-1));
    s[idx] = 0; idx += WARP_SIZE; s[idx] = val;
    if(0<=maxlvl) s[idx] += s[idx-1];
    if(1<=maxlvl) s[idx] += s[idx-2];
    if(2<=maxlvl) s[idx] += s[idx-4];
    if(3<=maxlvl) s[idx] += s[idx-8];
    if(4<=maxlvl) s[idx] += s[idx-16];
    return s[idx] - val;
}

extern "C" __global__ void kern_split_scan(const unsigned int* in, unsigned int* out, int N) {
    __shared__ volatile unsigned int sdata[512];
    int tid = threadIdx.x;
    unsigned int val = (tid < N) ? in[tid] : 0;
    unsigned int scanned = scanwarp_v(val, sdata, 4);
    __syncthreads();
    if (tid < N) out[tid] = scanned;
}

int main() {
    constexpr int N=256;
    unsigned int hi[N], ho[N];
    for(int i=0;i<N;i++) hi[i]=1;
    unsigned int*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_split_scan<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    // Warp-scan of all 1s: out[i] = i within warp (0,1,2,...,31,0,1,...)
    bool ok=(ho[0]==0 && ho[1]==1 && ho[31]==31);
    printf(ok?"TEST PASSED\n":"TEST FAILED (ho[0]=%u ho[1]=%u ho[31]=%u)\n",ho[0],ho[1],ho[31]);
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
