#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void MU_softmax(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int base = (i / 8) * 8;
        int end = base + 8; if (end > N) end = N;
        float maxv = in[base];
        for (int j = base+1; j < end; j++) maxv = fmaxf(maxv, in[j]);
        float sum = 0.0f;
        for (int j = base; j < end; j++) sum += expf(in[j] - maxv);
        out[i] = expf(in[i] - maxv) / sum;
    }
}
int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)((i%7)-3) * 0.5f;
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    MU_softmax<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int base=(i/8)*8,end=base+8; if(end>N) end=N;
        float maxv=h_in[base]; for(int j=base+1;j<end;j++) maxv=fmaxf(maxv,h_in[j]);
        float sum=0; for(int j=base;j<end;j++) sum+=expf(h_in[j]-maxv);
        float exp_val=expf(h_in[i]-maxv)/sum;
        if(fabsf(h_out[i]-exp_val)>1e-4f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
