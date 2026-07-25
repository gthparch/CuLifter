extern "C" __global__ void DM_normalize(const float* vx, const float* vy, const float* vz,
    float* ox, float* oy, float* oz, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N) {
        float len = sqrtf(vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]+1e-8f);
        float inv = 1.0f/len;
        ox[i]=vx[i]*inv; oy[i]=vy[i]*inv; oz[i]=vz[i]*inv;
    }
}
