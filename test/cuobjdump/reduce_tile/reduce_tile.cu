// file: cg_tile4_reduce.cu
#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void tile4_reduce_kernel(const float* A, float* B, int N) {
    // thread block tile size = 4
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 0;
    if (gid < N) {
        x = A[gid];

        // tile.size() == 4, tile.thread_rank() 0~3
        for (int offset = tile4.size() / 2; offset > 0; offset /= 2) {
            float y = tile4.shfl_down(x, offset);
            if (tile4.thread_rank() < offset) {
                x += y;
            }
        }
    }
    // useless sync -> can we find this? 
    //tile4.sync();

    if (gid < N && tile4.thread_rank() == 0) {
        int tile_id = gid / 4;
        B[tile_id] = x;
    }
}

int main() {
    const int N = 16;  
    float h_A[N], h_B[N/4];

    for (int i = 0; i < N; i++) h_A[i] = 1.0f; 

    float *d_A, *d_B;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, (N/4) * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);


    tile4_reduce_kernel<<<1, 32>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, (N/4) * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Tile reductions:\n");
    for (int i = 0; i < N/4; i++) {
        printf("B[%d] = %f\n", i, h_B[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
