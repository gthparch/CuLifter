#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void DM_idw_safe(const float* x, const float* y,
    const float* vals, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float sum_w = 0.0f, sum_wv = 0.0f;
    for (int i = 0; i < N; i++) {
        if (i == tid) continue;
        float dx = x[tid] - x[i], dy = y[tid] - y[i];
        float d = sqrtf(dx*dx + dy*dy + 1e-8f);
        float w = 1.0f / d;
        sum_w += w; sum_wv += w * vals[i];
    }
    out[tid] = sum_wv / (sum_w + 1e-8f);
}

int main(int argc, char** argv) {
    constexpr int N = 64;
    float h_x[N], h_y[N], h_v[N], h_out[N];
    for (int i = 0; i < N; i++) { h_x[i]=(float)(i%8)*1.5f; h_y[i]=(float)(i/8)*1.5f; h_v[i]=(float)(i*7%100); }
    memset(h_out, 0, sizeof(h_out));
    float *d_x,*d_y,*d_v,*d_out;
    cudaMalloc(&d_x,N*4);cudaMalloc(&d_y,N*4);cudaMalloc(&d_v,N*4);cudaMalloc(&d_out,N*4);
    cudaMemcpy(d_x,h_x,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,h_v,N*4,cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,N*4);
    DM_idw_safe<<<(N+63)/64,64>>>(d_x,d_y,d_v,d_out,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,N*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,4,N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++) if(isnan(h_out[i])||isinf(h_out[i])){printf("FAIL NaN/Inf at %d\n",i);ok=false;break;}
    printf(ok?"TEST PASSED\n":"FAIL\n");
    cudaFree(d_x);cudaFree(d_y);cudaFree(d_v);cudaFree(d_out);
    return ok?0:1;
}
