#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" __global__
void ISA5a_brev_clz_popc(int* out_brev, int* out_clz, int* out_popc,
                          const unsigned int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int val = in[tid];
        out_brev[tid] = (int)__brev(val);
        out_clz[tid]  = __clz(val);
        out_popc[tid] = __popc(val);
    }
}

static unsigned int cpu_brev(unsigned int v) {
    unsigned int r = 0;
    for (int i = 0; i < 32; i++) { r = (r << 1) | (v & 1); v >>= 1; }
    return r;
}
static int cpu_clz(unsigned int v) {
    if (v == 0) return 32;
    int n = 0; while (!(v & 0x80000000u)) { n++; v <<= 1; }
    return n;
}
static int cpu_popc(unsigned int v) {
    int c = 0; while (v) { c += v & 1; v >>= 1; } return c;
}

int main(int argc, char** argv) {
    const int N = 64;
    unsigned int h_in[64];
    int h_brev[64], h_clz[64], h_popc[64];
    h_in[0] = 0; h_in[1] = 1; h_in[2] = 0x80000000u; h_in[3] = 0xFFFFFFFFu;
    for (int i = 4; i < N; i++) h_in[i] = (unsigned)(i * 12345679 + 1);

    unsigned int *d_in; int *d_brev, *d_clz, *d_popc;
    cudaMalloc(&d_in, N * sizeof(unsigned));
    cudaMalloc(&d_brev, N * sizeof(int));
    cudaMalloc(&d_clz, N * sizeof(int));
    cudaMalloc(&d_popc, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(unsigned), cudaMemcpyHostToDevice);

    ISA5a_brev_clz_popc<<<1, 64>>>(d_brev, d_clz, d_popc, d_in, N);
    cudaMemcpy(h_brev, d_brev, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clz, d_clz, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_popc, d_popc, N * sizeof(int), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_brev, sizeof(int), N, f);
        fwrite(h_clz, sizeof(int), N, f);
        fwrite(h_popc, sizeof(int), N, f);
        fclose(f);
    }

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int eb = (int)cpu_brev(h_in[i]);
        int ec = cpu_clz(h_in[i]);
        int ep = cpu_popc(h_in[i]);
        if (h_brev[i] != eb || h_clz[i] != ec || h_popc[i] != ep) {
            printf("FAIL at [%d]: brev=%d/%d clz=%d/%d popc=%d/%d\n",
                   i, h_brev[i], eb, h_clz[i], ec, h_popc[i], ep);
            pass = 0;
        }
    }

    printf(pass ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(d_in); cudaFree(d_brev); cudaFree(d_clz); cudaFree(d_popc);
    return pass ? 0 : 1;
}
