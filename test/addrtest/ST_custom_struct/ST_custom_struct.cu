struct Params { float scale; int offset; float bias; };

extern "C" __global__ void ST_custom_struct(const float* in, float* out, Params p, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i + p.offset] * p.scale + p.bias;
}
