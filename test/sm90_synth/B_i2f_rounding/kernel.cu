#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void B_i2f_round(const int* in, float* out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N) return;
    // Different int-to-float conversions with large values (rounding matters)
    int val = in[i];
    float rn = __int2float_rn(val);  // I2F.RN (default)
    float rz = __int2float_rz(val);  // I2F.RZ
    float ru = __int2float_ru(val);  // I2F.RP
    float rd = __int2float_rd(val);  // I2F.RM
    out[i*4+0] = rn;
    out[i*4+1] = rz;
    out[i*4+2] = ru;
    out[i*4+3] = rd;
}

int main() {
    constexpr int N=64;
    int hi[N]; float ho[N*4];
    // Use large ints where rounding matters
    for(int i=0;i<N;i++) hi[i]=16777217+i; // 2^24+1, can't represent exactly in float
    int*di; float*do_; cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    B_i2f_round<<<1,64>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb"); fwrite(ho,4,N*4,f); fclose(f);
    printf("TEST PASSED\n");
    cudaFree(di);cudaFree(do_); return 0;
}
