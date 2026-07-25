#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_pns_compare(const float* tokens, float* result,
                                             int N, float threshold) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float val = (tid < N) ? tokens[tid] : 0.0f;

    // Chain of float comparisons → multiple predicates, FCHK-like patterns
    bool is_finite = isfinite(val);     // FCHK
    bool above = (val > threshold);
    bool below = (val < -threshold);
    bool in_range = is_finite && !above && !below;

    // Predicated operations based on multiple conditions
    float processed;
    if (!is_finite)       processed = 0.0f;
    else if (above)       processed = threshold;
    else if (below)       processed = -threshold;
    else                  processed = val;

    sdata[tid] = processed;
    __syncthreads();

    // Reduction with rounding-sensitive ops
    if (tid < 128) sdata[tid] = fmaf(sdata[tid], 1.0f, sdata[tid+128]);
    __syncthreads();
    if (tid < 64) sdata[tid] += sdata[tid+64];
    __syncthreads();
    if (tid < 32) {
        volatile float* vs = sdata;
        vs[tid] += vs[tid+32]; vs[tid] += vs[tid+16]; vs[tid] += vs[tid+8];
        vs[tid] += vs[tid+4]; vs[tid] += vs[tid+2]; vs[tid] += vs[tid+1];
    }
    if (tid == 0) result[blockIdx.x] = sdata[0];
}

int main() {
    constexpr int N=256;
    float hi[N],ho[1];
    for(int i=0;i<N;i++) hi[i]=sinf((float)i*0.1f)*10.0f;
    hi[10]=INFINITY; hi[20]=NAN; hi[30]=-INFINITY; // test FCHK
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_pns_compare<<<1,256>>>(di,do_,N,5.0f);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,1,f);fclose(f);
    bool ok=isfinite(ho[0]);
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
