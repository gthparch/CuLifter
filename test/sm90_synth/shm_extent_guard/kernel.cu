// shm_extent_guard — shared-memory extent semantics for the x86 wrapper.
//
// The kernel caches `words` floats of the input in DYNAMIC shared memory
// (the cudnn "singleread" pattern that overflowed the old fixed 192KB
// wrapper array), then reduces a per-thread strided sum.
//
// Three extents matter:
//   192KB  (words=49152): the OLD wrapper array's exact size — always fine.
//   224KB  (words=57344): valid on sm_90 (227KB opt-in cap) — the old
//          wrapper silently corrupted the host heap here; the sized
//          payload (228KB carveout) must run it CORRECTLY.
//   256KB  (words=65536): beyond the carveout — impossible on hardware
//          (cudaFuncSetAttribute caps at 227KB, launch is refused), so the
//          wrapper guard must report SKIP_SHMEM, never garbage.
#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_shm_extent(float* out, const float* in, int words) {
    extern __shared__ float cache[];
    for (int i = threadIdx.x; i < words; i += blockDim.x)
        cache[i] = in[i];
    __syncthreads();
    float s = 0.f;
    for (int i = threadIdx.x; i < words; i += blockDim.x)
        s += cache[i];
    out[threadIdx.x] = s;
}

static bool run_case(int words, float* h_out /*64*/) {
    const int BD = 64;
    size_t shm = size_t(words) * 4;
    cudaError_t e = cudaFuncSetAttribute(
        kern_shm_extent, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    if (e != cudaSuccess) { printf("attr refused (%zu B): %s\n", shm, cudaGetErrorString(e)); return false; }
    float* d_in; float* d_out;
    cudaMalloc(&d_in, shm); cudaMalloc(&d_out, BD * 4);
    float* h_in = new float[words];
    for (int i = 0; i < words; i++) h_in[i] = float(i % 8);
    cudaMemcpy(d_in, h_in, shm, cudaMemcpyHostToDevice);
    kern_shm_extent<<<1, BD, shm>>>(d_out, d_in, words);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("launch refused (%zu B): %s\n", shm, cudaGetErrorString(e)); return false; }
    cudaMemcpy(h_out, d_out, BD * 4, cudaMemcpyDeviceToHost);
    delete[] h_in; cudaFree(d_in); cudaFree(d_out);
    return true;
}

int main() {
    float out[64];
    bool ok = true;
    for (int words : {49152, 57344}) {
        if (!run_case(words, out)) { ok = false; continue; }
        // i%8 with stride 64 ≡ 0 (mod 8): every element of thread t is t%8.
        for (int t = 0; t < 64 && ok; t++)
            ok = (out[t] == float(words / 64) * float(t % 8));
        printf("words=%d %s\n", words, ok ? "ok" : "MISMATCH");
    }
    // 256KB: beyond the sm_90 carveout — the attribute/launch must refuse.
    if (run_case(65536, out)) { printf("UNEXPECTED: 256KB launched\n"); ok = false; }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
