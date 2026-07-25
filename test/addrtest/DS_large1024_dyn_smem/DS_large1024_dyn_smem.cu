extern "C" __global__ void DS_large1024_dyn_smem(const float* in, float* out, int N) {
    extern __shared__ float s[];
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) s[threadIdx.x]=in[i];
    __syncthreads();
    if(i<N) out[i]=s[threadIdx.x]*2.0f;
}
