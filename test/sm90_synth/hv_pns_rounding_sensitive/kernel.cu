#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_pns_rounding(const float* a, const float* b,
                                              float* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid >= N) return;
    // Deliberately rounding-sensitive computation
    // The difference between RM/RP/RZ matters for the final result
    float x = a[tid], y = b[tid];
    float r1 = __fmaf_rz(x, y, -x*y);  // should be ~0 but rounding makes it nonzero
    float r2 = __fmaf_rn(x, y, -x*y);
    float r3 = __fadd_rz(x, -x);        // should be exactly 0 with RZ
    float r4 = __fmul_rz(x, 1.0f/x);   // should be ~1 with RZ

    // Combine in a way that amplifies rounding differences
    out[tid*4+0] = r1;
    out[tid*4+1] = r2;
    out[tid*4+2] = r3;
    out[tid*4+3] = r4;
}

int main() {
    constexpr int N=64;
    float ha[N],hb[N],ho[N*4];
    for(int i=0;i<N;i++){ha[i]=1.0000001f+(float)i*1e-7f;hb[i]=0.9999999f+(float)i*1e-7f;}
    float*da,*db,*do_;
    cudaMalloc(&da,N*4);cudaMalloc(&db,N*4);cudaMalloc(&do_,N*4*4);
    cudaMemcpy(da,ha,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,N*4,cudaMemcpyHostToDevice);
    kern_pns_rounding<<<1,64>>>(da,db,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N*4,f);fclose(f);
    bool ok=!isnanf(ho[0]);
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(da);cudaFree(db);cudaFree(do_); return ok?0:1;
}
