#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

struct Vec2 { float x, y; };
struct Params { Vec2 origin; Vec2 scale; float* out; int N; };

extern "C" __global__ void kern_arg_nested(Params p) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=p.N) return;
    float fx = p.origin.x + p.scale.x * (float)tid;
    float fy = p.origin.y + p.scale.y * (float)tid;
    p.out[tid] = fx + fy;
}

int main() {
    constexpr int N=256;
    float ho[N]; float*do_;
    cudaMalloc(&do_,N*4);
    Params p = {{1.0f,2.0f},{0.1f,0.2f},do_,N};
    kern_arg_nested<<<1,256>>>(p);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    float exp=1.0f+2.0f; // tid=0: origin.x+origin.y
    bool ok=fabsf(ho[0]-exp)<1e-3f;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(do_); return ok?0:1;
}
