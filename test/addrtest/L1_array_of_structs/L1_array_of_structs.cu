struct MyStruct { float x; int y; float z; };

extern "C" __global__ void L1_array_of_structs(const MyStruct* arr, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = arr[i].x + (float)arr[i].y + arr[i].z;
}
