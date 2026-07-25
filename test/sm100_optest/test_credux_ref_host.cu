#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void test_credux(int* out, const int* in, int n) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (i < n) ? in[i] : 0;
    int sum = cg::reduce(tile, val, cg::plus<int>());
    if (tile.thread_rank() == 0) atomicAdd(out, sum);
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    bool dump = (argc > 1 && strcmp(argv[1], "--dump") == 0);

    int *h_in = new int[N], h_out = 0;
    for (int i = 0; i < N; i++) h_in[i] = i + 1;

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));

    test_credux<<<4, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (dump) {
        FILE* f = fopen("test_credux_ref_output.bin", "wb");
        fwrite(&h_out, sizeof(int), 1, f);
        fclose(f);
        printf("Dumped 1 int to test_credux_ref_output.bin\n");
    }

    int expected = N * (N + 1) / 2;  // sum of 1..N
    int errors = (h_out != expected) ? 1 : 0;
    if (errors) fprintf(stderr, "FAIL: got %d expected %d\n", h_out, expected);
    printf("%s\n", errors == 0 ? "TEST PASSED" : "FAIL");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in;
    return errors ? 1 : 0;
}
