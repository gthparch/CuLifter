__device__ int d_counter;

extern "C" __global__ void CB4b_device_int_rw(int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_counter = i * 2;  // write to device var
        out[i] = d_counter; // read back
    }
}
