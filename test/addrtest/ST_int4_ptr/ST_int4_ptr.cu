extern "C" __global__ void ST_int4_ptr(const int4* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int4 v = in[i];
        out[i] = v.x + v.y + v.z + v.w;
    }
}
