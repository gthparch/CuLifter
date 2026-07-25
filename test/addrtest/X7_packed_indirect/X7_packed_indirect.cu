#pragma pack(push, 1)
struct PackedEntry { char tag; float* ptr; short len; };
#pragma pack(pop)

extern "C" __global__ void X7_packed_indirect(const PackedEntry* entries,
    float* out, int N, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && i < entries[which].len)
        out[i] = entries[which].ptr[i];
}
