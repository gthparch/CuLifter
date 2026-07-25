// cluster_ctarank — SM90 SR_CgaCtaId (PTX %cluster_ctarank) microbench.
//
// Mirror of the HeCBench lda-cuda ReduceSum pattern: a static
// __shared__ float[32] scratch inside a multi-warp block reduction with
// __activemask() shuffles, alongside dynamic extern shared memory.
//
// NO clusters appear in source, yet ptxas -arch=sm_90 (12.2; sm_90a
// identical) addresses the static shared array through the
// shared::cluster window:
//
//     S2UR UR5, SR_CgaCtaId           (uniform read of the CTA's rank)
//     @!P0 S2R R12, SR_CgaCtaId       (predicated per-thread read)
//     LEA  R17, R17, 0x400, 0x18      addr = (rank << 24) + 0x400 + off
//
// For non-cluster launches the rank is 0 at runtime, so:
//   x86 backend:  constant 0 is exact (wrapper models cluster dims 1,1,1).
//   nvvm backend: must emit the real %cluster_ctarank sreg. Folding it to
//     constant 0 lets LLVM restructure the divergent regions around the
//     vote.sync calls, and the resulting valid, -O0-accepted PTX
//     SIGSEGVs ptxas 12.2 at -O1+.
#define WARP_SIZE 32

__inline__ __device__ float warp_reduce_sum(float val) {
  unsigned int active = __activemask();
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(active, val, offset);
  return val;
}

__inline__ __device__ float ReduceSum(const float* vec, const int length) {
  __shared__ float shared[32];
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  float val = 0.0f;
  for (int i = threadIdx.x; i < length; i += blockDim.x) val += vec[i];
  val = warp_reduce_sum(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  if (blockDim.x <= WARP_SIZE) return shared[0];
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
  if (warp == 0) {
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0) shared[0] = val;
  }
  __syncthreads();
  return shared[0];
}

extern "C" __global__ void kern_cluster_ctarank(float* out, const float* in,
                                                int n) {
  extern __shared__ float dyn[];
  for (int i = threadIdx.x; i < n; i += blockDim.x) dyn[i] = in[i];
  __syncthreads();
  float s = ReduceSum(dyn, n);
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}
