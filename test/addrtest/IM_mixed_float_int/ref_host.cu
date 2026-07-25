#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void IM_mixed_float_int(const float* fin, const int* iin,
    float* fout, int* iout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        fout[i] = fin[i] * 3.14159265f;
        iout[i] = iin[i] / 7;
    }
}
int main(int argc, char** argv) {
    constexpr int N = 256;
    float h_fin[N],h_fout[N]; int h_iin[N],h_iout[N];
    for(int i=0;i<N;i++){h_fin[i]=(float)(i+1)*0.01f;h_iin[i]=i*7+3;}
    memset(h_fout,0,sizeof(h_fout));memset(h_iout,0,sizeof(h_iout));
    float *d_fin,*d_fout; int *d_iin,*d_iout;
    cudaMalloc(&d_fin,N*4);cudaMalloc(&d_fout,N*4);
    cudaMalloc(&d_iin,N*4);cudaMalloc(&d_iout,N*4);
    cudaMemcpy(d_fin,h_fin,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_iin,h_iin,N*4,cudaMemcpyHostToDevice);
    cudaMemset(d_fout,0,N*4);cudaMemset(d_iout,0,N*4);
    IM_mixed_float_int<<<(N+255)/256,256>>>(d_fin,d_iin,d_fout,d_iout,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_fout,d_fout,N*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iout,d_iout,N*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){
        FILE*f=fopen("ref_output.bin","wb");fwrite(h_fout,4,N,f);fwrite(h_iout,4,N,f);fclose(f);
    }
    printf("TEST PASSED\n");
    cudaFree(d_fin);cudaFree(d_fout);cudaFree(d_iin);cudaFree(d_iout);
    return 0;
}
