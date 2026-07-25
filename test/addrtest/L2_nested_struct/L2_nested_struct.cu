struct Inner { float a; float b; };
struct Outer { int id; Inner inner; float c; };

extern "C" __global__ void L2_nested_struct(const Outer* arr, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = arr[i].inner.a + arr[i].inner.b + arr[i].c;
}
