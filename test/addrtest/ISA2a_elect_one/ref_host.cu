#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" __global__
void ISA2a_elect_one(int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int is_elected;
        asm volatile(
            "{\n\t"
            "  .reg .pred p;\n\t"
            "  elect.sync _|p, 0xFFFFFFFF;\n\t"
            "  selp.s32 %0, 1, 0, p;\n\t"
            "}\n\t"
            : "=r"(is_elected)
        );
        if (is_elected) {
            out[0] = tid;
        }
    }
}

int main(int argc, char** argv) {
    int h_out[1] = {-1};
    int *d_out;
    cudaMalloc(&d_out, sizeof(int));
    cudaMemset(d_out, 0xFF, sizeof(int));

    ISA2a_elect_one<<<1, 32>>>(d_out, 32);
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), 1, f);
        fclose(f);
    }

    if (h_out[0] == 0) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED: got %d, expected 0\n", h_out[0]);
        return 1;
    }
    cudaFree(d_out);
    return 0;
}
