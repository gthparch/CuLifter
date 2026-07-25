__global__ void dot_product(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ block_sums,
                            int N)
{
    extern __shared__ float sdata[];

    int tid    = threadIdx.x;
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        sum += A[i] * B[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // block reduction (simple tree)
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}