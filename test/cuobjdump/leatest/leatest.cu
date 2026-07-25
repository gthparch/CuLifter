__global__ void minimal_address_calculation(float* C, int N) {
    // 1. Calculate a 2D-like index from 1D thread/block IDs.
    // 'col' will be treated as a 32-bit integer offset.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 'row' is a simple index.
    int row = blockIdx.y;
    // 2. Force 64-bit pointer arithmetic.
    // The cast to 'long long' creates a 64-bit intermediate offset.
    long long row_offset = (long long)row * N;
    // 3. The critical expression that generates the LEA/LEA.HI.X pattern.
    // This is effectively: final_ptr64 = (base_ptr64 + offset64) + offset32
    float* ptr = C + row_offset + col;
    // 4. Use the pointer to prevent the calculation from being optimized away.
    *ptr = (float)col;
}