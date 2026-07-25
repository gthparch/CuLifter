#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

__device__ __forceinline__ unsigned int device_bfe(unsigned int val, unsigned int pos, unsigned int len) {
    unsigned int result;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(val), "r"(pos), "r"(len));
    return result;
}

__device__ __forceinline__ unsigned int device_bfi(unsigned int base, unsigned int insert, unsigned int pos, unsigned int len) {
    unsigned int result;
    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(result) : "r"(insert), "r"(base), "r"(pos), "r"(len));
    return result;
}

extern "C" __global__
void ISA5b_bfe_bfi(unsigned int* out, const unsigned int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int val = in[tid];
        unsigned int extracted = device_bfe(val, 8, 8);
        unsigned int inserted = device_bfi(0u, extracted, 16, 8);
        out[tid] = inserted;
    }
}

int main(int argc, char** argv) {
    const int N = 64;
    unsigned int h_in[64], h_out[64];
    for (int i = 0; i < N; i++) h_in[i] = 0x12345678u + i * 0x01010101u;

    unsigned int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(unsigned));
    cudaMalloc(&d_out, N * sizeof(unsigned));
    cudaMemcpy(d_in, h_in, N * sizeof(unsigned), cudaMemcpyHostToDevice);

    ISA5b_bfe_bfi<<<1, 64>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, N * sizeof(unsigned), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(unsigned), N, f);
        fclose(f);
    }

    int pass = 1;
    for (int i = 0; i < N; i++) {
        unsigned int byte1 = (h_in[i] >> 8) & 0xFF;
        unsigned int expected = byte1 << 16;
        if (h_out[i] != expected) {
            printf("FAIL [%d]: got 0x%08x, expected 0x%08x\n", i, h_out[i], expected);
            pass = 0;
        }
    }

    printf(pass ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(d_in); cudaFree(d_out);
    return pass ? 0 : 1;
}
