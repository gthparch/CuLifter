// Multi-tile UTMALDG.4D microbench (the FA 4D-coordinate-recovery test).
//
// cute folds 2D-tensor tile slices into the descriptor (always emits .2D), and
// its rank>=3 partition path hits static_asserts for a rank-2 smem layout —
// the wall every prior agent hit. So this microbench drives the genuine
// `cp.async.bulk.tensor.4d` PTX directly with a host-encoded rank-4 CUtensorMap
// (cuTensorMapEncodeTiled), exactly the cute SM90_TMA_LOAD ABI. The kernel loops
// the M (seq) tile coordinate, so each iteration issues one UTMALDG.4D with the
// SAME descriptor but a DIFFERENT coordinate vector {0, m_tile, 0, 0} — the
// coords land in the uniform-register window (UR10..UR13) that the x86 lifter
// must recover by dataflow. Each loaded tile is stored straight back to global,
// so the host verifies tile `it` equals global rows [it*M_TILE, +M_TILE).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

static constexpr int DIM    = 64;   // contiguous box dim (mode 0)
static constexpr int M_TILE = 8;    // rows per tile (box dim, mode 1)
static constexpr int NTILE  = 4;    // number of M-tiles (coords vary)
static constexpr int SEQ    = M_TILE * NTILE;  // 32
static constexpr int H2     = 1;    // modes 2,3 are size-1 (still rank-4 -> .4D)
static constexpr int H3     = 1;

// PTX cp.async.bulk.tensor.4d wrapper. tmap = __grid_constant__ CUtensorMap.
__device__ __forceinline__ void tma_load_4d(void* smem, const CUtensorMap* tmap,
                                            uint64_t* mbar,
                                            int c0, int c1, int c2, int c3) {
  uint32_t s_smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  uint32_t s_mbar = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4, %5, %6}], [%2];"
    :: "r"(s_smem), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(s_mbar),
       "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    : "memory");
}

__device__ __forceinline__ void mbar_init(uint64_t* mbar, int count) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(s), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* mbar, int bytes) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(s), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, int phase) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
    "@p bra DONE;\n"
    "bra LAB_WAIT;\n"
    "DONE:\n"
    "}\n" :: "r"(s), "r"(phase) : "memory");
}

extern "C" __global__ void
tma_4d_multitile_kernel(const __grid_constant__ CUtensorMap tmap,
                        half* __restrict__ gOut) {
#if defined(__CUDA_ARCH__)
  extern __shared__ char smem_raw[];
  half* sA = reinterpret_cast<half*>(smem_raw);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw + sizeof(half) * M_TILE * DIM + 1024);
  int t = threadIdx.x;
  constexpr int kBytes = sizeof(half) * M_TILE * DIM;

  for (int it = 0; it < NTILE; ++it) {
    if (t == 0) {
      mbar_init(mbar, 1);
      mbar_expect(mbar, kBytes);
      // coords {mode0=DIM-offset(0), mode1=seq-tile-row, mode2=0, mode3=0}.
      tma_load_4d(sA, &tmap, mbar, 0, it * M_TILE, 0, 0);
    }
    __syncthreads();
    mbar_wait(mbar, 0);
    __syncthreads();

    // store this tile back to gOut at its tile rows (linear K-major in smem).
    for (int i = t; i < M_TILE * DIM; i += blockDim.x) {
      int m = i / DIM, k = i % DIM;
      gOut[(it * M_TILE + m) * DIM + k] = sA[m * DIM + k];
    }
    __syncthreads();
  }
#endif
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s OUTPUT_DIR\n", argv[0]);
    return 2;
  }
  const char* OUT = argv[1];
  CUresult cr = cuInit(0);
  CUdevice dev; cuDeviceGet(&dev, 0);
  CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

  half hA[SEQ * DIM], hOut[SEQ * DIM];
  for (int i = 0; i < SEQ * DIM; i++)
    hA[i] = __float2half(((i * 37 % 101) - 50) * 0.02f);
  memset(hOut, 0, sizeof(hOut));

  half *dA, *dOut;
  cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dOut, sizeof(hOut));
  cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
  cudaMemset(dOut, 0, sizeof(hOut));

  // Encode a rank-4 CUtensorMap over the (DIM, SEQ, 1, 1) tensor (K-major:
  // mode 0 = DIM contiguous). Box = (DIM, M_TILE, 1, 1) so each TMA grabs one
  // M_TILE x DIM tile; the seq coordinate (mode 1) selects the tile.
  CUtensorMap tmap{};
  uint64_t globalDim[4]    = { (uint64_t)DIM, (uint64_t)SEQ, 1, 1 };
  // strides in BYTES of modes 1..3 (mode 0 stride is implicit = elem size).
  uint64_t globalStride[3] = { (uint64_t)DIM * sizeof(half),
                               (uint64_t)DIM * SEQ * sizeof(half),
                               (uint64_t)DIM * SEQ * sizeof(half) };
  uint32_t boxDim[4]       = { (uint32_t)DIM, (uint32_t)M_TILE, 1, 1 };
  uint32_t elemStride[4]   = { 1, 1, 1, 1 };
  cr = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, dA, globalDim, globalStride,
      boxDim, elemStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (cr != CUDA_SUCCESS) {
    const char* s; cuGetErrorString(cr, &s);
    printf("cuTensorMapEncodeTiled failed: %s\n", s); return 1;
  }

  int smem_bytes = sizeof(half) * M_TILE * DIM + 2048;
  cudaFuncSetAttribute(tma_4d_multitile_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  tma_4d_multitile_kernel<<<1, 128, smem_bytes>>>(tmap, dOut);
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hOut, dOut, sizeof(hOut), cudaMemcpyDeviceToHost);

  double maxabs = 0;
  for (int i = 0; i < SEQ * DIM; i++) {
    double diff = fabs((double)__half2float(hOut[i]) - (double)__half2float(hA[i]));
    if (diff > maxabs) maxabs = diff;
  }
  printf("=== TMA 4D multitile out[0..7] ===\n");
  for (int k = 0; k < 8; k++) printf("%.4f ", __half2float(hOut[k])); printf("\n");
  printf("=== row M_TILE (tile 1) out ===\n");
  for (int k = 0; k < 8; k++) printf("%.4f ", __half2float(hOut[M_TILE * DIM + k])); printf("\n");
  printf("max |out - in| = %.5f\n", maxabs);

  char p[256];
  snprintf(p, sizeof(p), "%s/in_A.bin", OUT);  { FILE* f = fopen(p, "wb"); fwrite(hA, sizeof(half), SEQ * DIM, f); fclose(f); }
  snprintf(p, sizeof(p), "%s/ref_out.bin", OUT); { FILE* f = fopen(p, "wb"); fwrite(hOut, sizeof(half), SEQ * DIM, f); fclose(f); }
  if (maxabs < 1e-3) { printf("TEST PASSED\n"); return 0; }
  printf("TEST FAILED\n");
  return 1;
}
