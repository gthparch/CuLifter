#include <cuda/pipeline>
#include <cstdio>
#include <cstdint>

// LDGSTS 128-bit: async copy 16 bytes (4 floats) from global to shared per thread
extern "C" __global__ void ldgsts_128(const float4* __restrict__ in, float4* out, int N) {
    __shared__ float4 smem[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    if (tid < N)
        cuda::memcpy_async(&smem[threadIdx.x], &in[tid], sizeof(float4), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    if (tid < N) {
        float4 v = smem[threadIdx.x];
        v.x += 1.0f;
        v.y += 1.0f;
        v.z += 1.0f;
        v.w += 1.0f;
        out[tid] = v;
    }
}

int main() {
    constexpr int N = 256;
    float4 h_in[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = make_float4((float)(4*i+1), (float)(4*i+2), (float)(4*i+3), (float)(4*i+4));
    }

    float4 *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float4));
    cudaMalloc(&d_out, N * sizeof(float4));
    cudaMemcpy(d_in, h_in, N * sizeof(float4), cudaMemcpyHostToDevice);

    ldgsts_128<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float4), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float4), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float ex = (float)(4*i+1) + 1.0f;
        float ey = (float)(4*i+2) + 1.0f;
        float ez = (float)(4*i+3) + 1.0f;
        float ew = (float)(4*i+4) + 1.0f;
        if (fabsf(h_out[i].x - ex) > 1e-5f || fabsf(h_out[i].y - ey) > 1e-5f ||
            fabsf(h_out[i].z - ez) > 1e-5f || fabsf(h_out[i].w - ew) > 1e-5f) {
            printf("MISMATCH at %d\n", i);
            ok = false;
            break;
        }
    }
    if (ok) printf("GPU REF PASSED\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
}
