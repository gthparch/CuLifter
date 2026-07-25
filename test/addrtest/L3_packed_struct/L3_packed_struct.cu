#pragma pack(push, 1)
struct Packed { char a; float b; short c; float d; };
#pragma pack(pop)

extern "C" __global__ void L3_packed_struct(const Packed* arr, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = arr[i].b + arr[i].d + (float)arr[i].c;
}
