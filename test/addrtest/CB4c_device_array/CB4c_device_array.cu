__device__ int d_arr[256];

extern "C" __global__ void CB4c_device_array(const int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_arr[i] = in[i] * 3;
        out[i] = d_arr[i];
    }
}
