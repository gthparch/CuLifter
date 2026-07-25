#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Reproduce quicksort's lqsort generic-shared-pointer round-trip: large
// __shared__ arrays mys/mysn, __shared__ generic pointers s/sn selected by a
// `direction` read from a __shared__ workstack record, then deref s/sn through
// the reloaded pointer. Exercises the SR_SWINHI generic-pointer construction.
#define WG 128
#define BLK 1024

struct rec { unsigned start, end, direction; };

extern "C" __global__ void kern_shared_genptr(unsigned* d, unsigned* dn, rec* seqs) {
    __shared__ unsigned mys[BLK];
    __shared__ unsigned mysn[BLK];
    __shared__ unsigned* s;
    __shared__ unsigned* sn;
    __shared__ rec ws;

    unsigned localid = threadIdx.x;
    if (localid == 0) ws = seqs[blockIdx.x];
    __syncthreads();

    for (unsigned i = localid; i < BLK; i += WG) { mys[i] = d[i]; mysn[i] = d[i] + 100000; }
    __syncthreads();

    if (localid == 0) {
        if (ws.direction == 1) { s = mys;  sn = mysn; }
        else                   { s = mysn; sn = mys;  }
    }
    __syncthreads();

    // deref through the reloaded generic pointers across a range of indices
    unsigned acc = 0;
    for (unsigned i = ws.start + localid; i < ws.end; i += WG) {
        acc += s[i] + sn[i];
    }
    // reduce per-block via global atomics into dn[blockIdx.x]
    atomicAdd(&dn[blockIdx.x], acc);
}

int main() {
    constexpr int BLOCKS = 256, N = BLK;
    rec hseq[BLOCKS];
    unsigned *hd = new unsigned[N], *hdn = new unsigned[BLOCKS];
    for (int i = 0; i < N; i++) hd[i] = (unsigned)(i * 7 + 3);
    for (int b = 0; b < BLOCKS; b++) { hseq[b].start = 0; hseq[b].end = BLK; hseq[b].direction = (b & 1); }

    unsigned *dd, *ddn; rec* dseq;
    cudaMalloc(&dd, N * 4); cudaMalloc(&ddn, BLOCKS * 4);
    cudaMalloc(&dseq, BLOCKS * sizeof(rec));
    cudaMemcpy(dd, hd, N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dseq, hseq, BLOCKS * sizeof(rec), cudaMemcpyHostToDevice);
    cudaMemset(ddn, 0, BLOCKS * 4);

    kern_shared_genptr<<<dim3(BLOCKS), dim3(WG)>>>(dd, ddn, dseq);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("TEST FAILED (launch: %s)\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(hdn, ddn, BLOCKS * 4, cudaMemcpyDeviceToHost);

    // per block: sum over i of (mys[i] + mysn[i]) = sum (hd[i] + hd[i]+100000)
    unsigned exp = 0;
    for (int i = 0; i < N; i++) exp += hd[i] + (hd[i] + 100000);
    bool ok = true;
    for (int b = 0; b < BLOCKS; b++)
        if (hdn[b] != exp) { printf("block %d: got %u want %u\n", b, hdn[b], exp); ok = false; break; }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
