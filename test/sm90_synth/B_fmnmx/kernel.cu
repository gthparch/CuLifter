#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void B_fmnmx(const float* a, const float* b, float* out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N) return;
    float mn = fminf(a[i], b[i]);         // FMNMX (min)
    float mx = fmaxf(a[i], b[i]);         // FMNMX (max)
    float clamped = fminf(fmaxf(a[i], 0.0f), 1.0f); // clamp [0,1]
    out[i*3+0] = mn;
    out[i*3+1] = mx;
    out[i*3+2] = clamped;
}

int main() {
    constexpr int N=128;
    float ha[N],hb[N],ho[N*3];
    for(int i=0;i<N;i++){ha[i]=(float)(i-64)*0.1f;hb[i]=(float)(64-i)*0.1f;}
    float*da,*db,*do_; cudaMalloc(&da,N*4);cudaMalloc(&db,N*4);cudaMalloc(&do_,N*3*4);
    cudaMemcpy(da,ha,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,N*4,cudaMemcpyHostToDevice);
    B_fmnmx<<<1,128>>>(da,db,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*3*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb"); fwrite(ho,4,N*3,f); fclose(f);
    // i=0: a=-6.4, b=6.4 → min=-6.4, max=6.4, clamp=0.0
    bool ok=(ho[0]==-6.4f && ho[1]==6.4f && ho[2]==0.0f);
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(da);cudaFree(db);cudaFree(do_); return ok?0:1;
}
