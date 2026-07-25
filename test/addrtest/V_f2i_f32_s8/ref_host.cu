#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void V_f2i_f32_s8(const float* in, signed char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (signed char)((signed char)roundf(in[i]));
}
int main(int argc, char** argv) {
    constexpr int N=1024;
    float h_in[N]; signed char h_out[N];
    for(int i=0;i<N;i++) h_in[i]=(float)((i%200)-100);
    memset(h_out,0,sizeof(h_out));
    float *d_in; signed char *d_out;
    cudaMalloc(&d_in,N*sizeof(float));cudaMalloc(&d_out,N*1);
    cudaMemcpy(d_in,h_in,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,N*1);
    V_f2i_f32_s8<<<(N+255)/256,256>>>(d_in,d_out,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,N*1,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,1,N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++){
        signed char exp=(signed char)((signed char)roundf(h_in[i]));
        if(h_out[i]!=exp){printf("FAIL %d\n",i);ok=false;break;}
    }
    printf(ok?"TEST PASSED\n":"FAIL\n");
    cudaFree(d_in);cudaFree(d_out);
    return ok?0:1;
}
