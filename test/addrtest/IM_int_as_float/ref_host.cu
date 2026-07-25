#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void IM_int_as_float(const int* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float f = __int_as_float(in[i]);
        out[i] = f * 2.0f;
    }
}
int main(int argc, char** argv) {
    constexpr int N = 256;
    int h_in[N]; float h_out[N];
    for(int i=0;i<N;i++){float v=(float)(i+1)*0.1f;memcpy(&h_in[i],&v,4);}
    memset(h_out,0,sizeof(h_out));
    int *d_in; float *d_out;
    cudaMalloc(&d_in,N*4);cudaMalloc(&d_out,N*4);
    cudaMemcpy(d_in,h_in,N*4,cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,N*4);
    IM_int_as_float<<<(N+255)/256,256>>>(d_in,d_out,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,N*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,4,N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++){float v;memcpy(&v,&h_in[i],4);float exp=v*2.0f;
    if(fabsf(h_out[i]-exp)>fabsf(exp)*1e-5f+1e-7f){printf("FAIL %d\n",i);ok=false;break;}}
    printf(ok?"TEST PASSED\n":"FAIL\n");
    cudaFree(d_in);cudaFree(d_out);return ok?0:1;
}
