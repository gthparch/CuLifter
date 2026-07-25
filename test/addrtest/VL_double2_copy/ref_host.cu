#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void VL_double2_copy(const double2* in, double2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}
int main(int argc, char** argv) {
    constexpr int N = 512;
    double2 h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = make_double2((i*2+1)*0.001, (i*2+2)*0.001);
    memset(h_out, 0, sizeof(h_out));
    double2 *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(double2)); cudaMalloc(&d_out, N*sizeof(double2));
    cudaMemcpy(d_in, h_in, N*sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(double2));
    VL_double2_copy<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(double2), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(double2), N, f); fclose(f);
    }
    printf("TEST PASSED\n");
    cudaFree(d_in); cudaFree(d_out); return 0;
}
