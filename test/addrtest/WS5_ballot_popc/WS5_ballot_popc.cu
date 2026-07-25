extern "C" __global__ void WS5_ballot_popc(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < N) ? in[tid] : 0;
    unsigned mask = __ballot_sync(0xffffffff, val > 0);
    int count = __popc(mask);
    if (tid < N) out[tid] = count;
}
