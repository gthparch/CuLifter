#include <cstdio>
#include <cuda_runtime.h>
// Tests the warp-shift primitives used in btree insert_into_node
// (map_warps.cu): __shfl_down_sync(delta=2), __shfl_up_sync, and the
// branchless lane-masked select. These are SHFL.DOWN/SHFL.UP, untested by
// ballot_elect (broadcast) / pool4gb (addressing).
#define WM 0xFFFFFFFFu
__global__ void shift_kernel(const unsigned* in, unsigned* out) {
  int lane = threadIdx.x & 31;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned v = in[gid];
  unsigned down2 = __shfl_down_sync(WM, v, 2, 32);   // node "move up two spaces"
  unsigned up1   = __shfl_up_sync(WM, v, 1, 32);
  unsigned idx29 = __shfl_sync(WM, v, 29, 32);        // constant-lane reads btree uses
  unsigned idx31 = __shfl_sync(WM, v, 31, 32);
  unsigned to_move = (1u << ((lane + 5) & 31)) - 1u;
  bool shifted = ((1u << lane) & to_move) != 0;
  unsigned sel = shifted * down2 + (1u - shifted) * v;  // branchless select idiom
  out[gid] = sel ^ up1 ^ idx29 ^ idx31;
}
int main() {
  const int WGS = 256, BLOCKS = 8, N = WGS * BLOCKS;
  unsigned *in, *out; unsigned long long chk = 0;
  cudaMallocManaged(&in, N * sizeof(unsigned));
  cudaMallocManaged(&out, N * sizeof(unsigned));
  for (int i = 0; i < N; i++) in[i] = (unsigned)(i * 2654435761u + 1u);
  shift_kernel<<<BLOCKS, WGS>>>(in, out);
  cudaDeviceSynchronize();
  for (int i = 0; i < N; i++) chk = chk * 1000003u + out[i];
  printf("checksum=%llu\n", chk);
  return 0;
}
