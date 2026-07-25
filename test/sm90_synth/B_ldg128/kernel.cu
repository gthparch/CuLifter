#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_ldg128(const float4* __restrict__ in,
                                        float4* __restrict__ out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float4 v = in[tid];  // LDG.E.128 → R:R+1:R+2:R+3
    v.x *= 2.0f; v.y *= 2.0f; v.z *= 2.0f; v.w *= 2.0f;
    out[tid] = v;         // STG.E.128
}

int main() {
    constexpr int N = 256;
    float4 hi[N], ho[N];
    for (int i = 0; i < N; i++) {
        hi[i].x = (float)(i*4+1); hi[i].y = (float)(i*4+2);
        hi[i].z = (float)(i*4+3); hi[i].w = (float)(i*4+4);
    }
    float4 *di, *do_;
    cudaMalloc(&di, N*sizeof(float4)); cudaMalloc(&do_, N*sizeof(float4));
    cudaMemcpy(di, hi, N*sizeof(float4), cudaMemcpyHostToDevice);
    kern_ldg128<<<(N+255)/256, 256>>>(di, do_, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho, do_, N*sizeof(float4), cudaMemcpyDeviceToHost);
    FILE *f = fopen("ref_output.bin", "wb"); fwrite(ho, sizeof(float4), N, f); fclose(f);
    bool ok = (ho[0].x == 2.0f && ho[0].y == 4.0f && ho[0].z == 6.0f && ho[0].w == 8.0f);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(di); cudaFree(do_); return ok ? 0 : 1;
}
