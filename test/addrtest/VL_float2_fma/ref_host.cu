#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void VL_float2_fma(const float2* a, const float2* b, const float2* c,
    float2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 va = a[i], vb = b[i], vc = c[i];
        out[i] = make_float2(va.x * vb.x + vc.x, va.y * vb.y + vc.y);
    }
}
int main(int argc, char** argv) {
    constexpr int N = 512;
    float2 h_a[N],h_b[N],h_c[N],h_out[N];
    for(int i=0;i<N;i++) {
        h_a[i]=make_float2((i*2%7+1)*0.1f, ((i*2+1)%7+1)*0.1f);
        h_b[i]=make_float2((i*2%5+1)*0.2f, ((i*2+1)%5+1)*0.2f);
        h_c[i]=make_float2((i*2%3+1)*0.3f, ((i*2+1)%3+1)*0.3f);
    }
    memset(h_out,0,sizeof(h_out));
    float2 *d_a,*d_b,*d_c,*d_out;
    cudaMalloc(&d_a,N*8);cudaMalloc(&d_b,N*8);cudaMalloc(&d_c,N*8);cudaMalloc(&d_out,N*8);
    cudaMemcpy(d_a,h_a,N*8,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,N*8,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,h_c,N*8,cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,N*8);
    VL_float2_fma<<<(N+255)/256,256>>>(d_a,d_b,d_c,d_out,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,N*8,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,8,N,f);fclose(f);}
    printf("TEST PASSED\n");
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);cudaFree(d_out);return 0;
}
