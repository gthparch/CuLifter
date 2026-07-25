extern "C" __global__ void VL_int4_bitwise(const int4* a, const int4* b, int4* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int4 va = a[i], vb = b[i];
        out[i] = make_int4(va.x ^ vb.x, va.y & vb.y, va.z | vb.z, ~va.w);
    }
}
