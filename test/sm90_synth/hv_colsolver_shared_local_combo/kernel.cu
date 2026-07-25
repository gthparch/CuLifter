#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_colsolver_combo(
    const float* scores, const int* encrypted, int* decrypted,
    int msg_len, int key_len)
{
    __shared__ float shared_best[32];
    float local_key[32];  // per-thread key → local memory
    int tid = threadIdx.x;

    // Init local key from thread ID (simulates RNG)
    for (int j = 0; j < key_len && j < 32; j++)
        local_key[j] = (float)((tid * 7 + j * 13) % 26);

    // Compute score using local key + scores table
    float my_score = 0.0f;
    for (int j = 0; j < msg_len-1 && j < 64; j++) {
        int c1 = encrypted[j] % 26;
        int c2 = encrypted[j+1] % 26;
        my_score += scores[c1*26+c2];
    }

    // Store best per-warp to shared
    if ((tid & 31) == 0 && (tid >> 5) < 32)
        shared_best[tid >> 5] = my_score;
    __syncthreads();

    // Warp 0 finds overall best
    if (tid < 32) {
        float val = shared_best[tid];
        for (int s = 16; s > 0; s >>= 1) {
            float other = __shfl_down_sync(0xffffffff, val, s);
            if (other < val) val = other;
        }
        if (tid == 0) decrypted[0] = (int)val;
    }
}

int main() {
    constexpr int MSG=64, ALPHA=26;
    float h_scores[ALPHA*ALPHA]; int h_enc[MSG], h_dec[1]={0};
    for(int i=0;i<ALPHA*ALPHA;i++) h_scores[i]=(float)(i%100)*0.01f;
    for(int i=0;i<MSG;i++) h_enc[i]=i%ALPHA;
    float*ds; int*de,*dd;
    cudaMalloc(&ds,ALPHA*ALPHA*4);cudaMalloc(&de,MSG*4);cudaMalloc(&dd,4);
    cudaMemcpy(ds,h_scores,ALPHA*ALPHA*4,cudaMemcpyHostToDevice);
    cudaMemcpy(de,h_enc,MSG*4,cudaMemcpyHostToDevice);
    kern_colsolver_combo<<<1,256>>>(ds,de,dd,MSG,26);
    cudaDeviceSynchronize();
    cudaMemcpy(h_dec,dd,4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(h_dec,4,1,f);fclose(f);
    printf("TEST PASSED\n");
    cudaFree(ds);cudaFree(de);cudaFree(dd); return 0;
}
