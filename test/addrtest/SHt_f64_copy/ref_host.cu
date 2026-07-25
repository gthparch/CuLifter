#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void SHt_f64_copy(const double* in, double* out, int N) {
    __shared__ double s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x];
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    double h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (double)(i*3+7);
    memset(h_out, 0, sizeof(h_out));
    double *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(double)); cudaMalloc(&d_out, N*sizeof(double));
    cudaMemcpy(d_in, h_in, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(double));
    SHt_f64_copy<<<(N+BS-1)/BS, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(double), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(double), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabs(h_out[i]-h_in[i])>1e-10) { printf("FAIL at %d\n",i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
