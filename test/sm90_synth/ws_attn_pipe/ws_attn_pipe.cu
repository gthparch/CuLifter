// ws_attn_pipe — MINIMAL warp-specialized FlashAttention forward.
//
// The FA3 mainloop structure at tractable scale: a PRODUCER warp streams K/V
// tiles into a 2-stage shared ring buffer guarded by full[]/empty[] mbarriers,
// while a CONSUMER warp runs online-softmax attention out of the ring. This is
// genuine warp specialization (producer/consumer warps + multi-stage mbarrier
// handshake + phase reuse across KV tiles) but uses scalar FMA compute (no
// WGMMA -> no x86 perf wall) and flat pointer args (no CUTLASS Params), so it
// can actually run + be validated on x86. Single CTA, 2 warps (64 threads).
//
//   Q[SQ][D], K[SK][D], V[SK][D], O[SQ][D] (row-major f32); consumer lane i owns
//   query row i. out[i] = softmax(scale * Q_i . K^T) . V.
#include <cstdint>
#include <cuda.h>

#define D       16
#define SQ      32
#define KTILE   32
#define NSTAGE  2

__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "l"(b), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "l"(b));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
    asm volatile("{\n\t.reg .pred P;\n\tLW:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                 "@!P bra LW;\n\t}\n\t" :: "l"(b), "r"(parity));
}

extern "C" __global__ void ws_attn_pipe(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        float* __restrict__ O, int seqlen_k) {
    __shared__ float sK[NSTAGE][KTILE][D];
    __shared__ float sV[NSTAGE][KTILE][D];
    __shared__ __align__(8) uint64_t full[NSTAGE];
    __shared__ __align__(8) uint64_t empty[NSTAGE];

    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int n_tiles = seqlen_k / KTILE;
    const float scale = 0.25f;   // 1/sqrt(D=16)

    if (tid == 0)
        for (int s = 0; s < NSTAGE; s++) { mbar_init(&full[s], 32); mbar_init(&empty[s], 32); }
    __syncthreads();

    if (warp == 0) {
        // PRODUCER: stream K/V tiles into the ring.
        for (int t = 0; t < n_tiles; t++) {
            int s = t % NSTAGE;
            uint32_t ph = (uint32_t)((t / NSTAGE) & 1);
            mbar_wait(&empty[s], ph);                 // slot free?
            for (int k = 0; k < D; k++) {
                sK[s][lane][k] = K[(t * KTILE + lane) * D + k];
                sV[s][lane][k] = V[(t * KTILE + lane) * D + k];
            }
            mbar_arrive(&full[s]);                     // data ready
        }
    } else {
        // CONSUMER: prime empty[] (slots start free), then online-softmax attn.
        for (int s = 0; s < NSTAGE; s++) mbar_arrive(&empty[s]);

        float q[D];
        for (int k = 0; k < D; k++) q[k] = Q[lane * D + k];
        float m = -1e30f, l = 0.0f, acc[D];
        for (int k = 0; k < D; k++) acc[k] = 0.0f;

        for (int t = 0; t < n_tiles; t++) {
            int s = t % NSTAGE;
            uint32_t ph = (uint32_t)((t / NSTAGE) & 1);
            mbar_wait(&full[s], ph);                   // data ready?

            float sc[KTILE], tmax = -1e30f;
            for (int j = 0; j < KTILE; j++) {
                float dot = 0.0f;
                for (int k = 0; k < D; k++) dot += q[k] * sK[s][j][k];
                sc[j] = dot * scale;
                tmax = fmaxf(tmax, sc[j]);
            }
            float m_new = fmaxf(m, tmax);
            float corr = __expf(m - m_new);
            l *= corr;
            for (int k = 0; k < D; k++) acc[k] *= corr;
            for (int j = 0; j < KTILE; j++) {
                float p = __expf(sc[j] - m_new);
                l += p;
                for (int k = 0; k < D; k++) acc[k] += p * sV[s][j][k];
            }
            m = m_new;

            mbar_arrive(&empty[s]);                    // slot free
        }
        float inv = 1.0f / l;
        for (int k = 0; k < D; k++) O[lane * D + k] = acc[k] * inv;
    }
}

int main() {
    constexpr int SK = 64;
    float Q[SQ * D], K[SK * D], Vv[SK * D], O[SQ * D];
    for (int i = 0; i < SQ * D; i++) Q[i]  = (float)(((i * 7 + 5) % 13) - 6) * 0.1f;
    for (int i = 0; i < SK * D; i++) K[i]  = (float)(((i * 5 + 3) % 11) - 5) * 0.1f;
    for (int i = 0; i < SK * D; i++) Vv[i] = (float)(((i * 3 + 1) % 9)  - 4) * 0.1f;
    float *dQ,*dK,*dV,*dO; cudaMalloc(&dQ,sizeof Q);cudaMalloc(&dK,sizeof K);cudaMalloc(&dV,sizeof Vv);cudaMalloc(&dO,sizeof O);
    cudaMemcpy(dQ,Q,sizeof Q,cudaMemcpyHostToDevice);cudaMemcpy(dK,K,sizeof K,cudaMemcpyHostToDevice);cudaMemcpy(dV,Vv,sizeof Vv,cudaMemcpyHostToDevice);
    ws_attn_pipe<<<1,64>>>(dQ,dK,dV,dO,SK);
    cudaDeviceSynchronize(); cudaMemcpy(O,dO,sizeof O,cudaMemcpyDeviceToHost);
    FILE* f=fopen("ref_output.bin","wb"); fwrite(O,4,SQ*D,f); fclose(f);
    const float scale=0.25f; bool ok=true; float maxerr=0;
    for (int i=0;i<SQ&&ok;i++){ float s[SK],mx=-1e30f;
        for(int j=0;j<SK;j++){float d=0;for(int k=0;k<D;k++)d+=Q[i*D+k]*K[j*D+k];s[j]=d*scale;if(s[j]>mx)mx=s[j];}
        float den=0;for(int j=0;j<SK;j++){s[j]=expf(s[j]-mx);den+=s[j];}
        for(int k=0;k<D;k++){float o=0;for(int j=0;j<SK;j++)o+=s[j]*Vv[j*D+k];o/=den;float e=fabsf(o-O[i*D+k]);if(e>maxerr)maxerr=e;if(e>1e-3f){printf("MISMATCH %d %d got %f exp %f\n",i,k,O[i*D+k],o);ok=false;break;}}}
    printf("max_err=%g\n",maxerr); printf(ok?"GPU REF PASSED\n":"GPU REF FAILED\n");
    return ok?0:1;
}
