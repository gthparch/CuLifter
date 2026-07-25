struct Vec3 { float x, y, z; };

extern "C" __global__ void L5_struct_write(const float* in, Vec3* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i].x = in[i] * 1.0f;
        out[i].y = in[i] * 2.0f;
        out[i].z = in[i] * 3.0f;
    }
}
