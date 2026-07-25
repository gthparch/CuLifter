// Microbenchmark: cub DeviceRadixSortOnesweepKernel PARTIAL-TILE path
// (HeCBench sa-cuda / sortKV-cuda residual blocker).
//
// Instantiates the EXACT kernel the sa-cuda bench SASS contains:
//   DeviceRadixSortOnesweepKernel<policy_hub<int,int,unsigned long long>::Policy1000,
//                                 SortOrder::Ascending, int, int,
//                                 unsigned long long, int, int,
//                                 identity_decomposer_t>
// (bench = CUB 3.0.104 / CUDA 13.0; here = CUB 3.0.200 / CUDA 13.2 — same
//  agent family). Driving it with num_items=32 while TILE_ITEMS=1536 takes
//  the full_block=false path: guarded LoadDirectWarpStriped with DefaultKey
//  fill, ranking over real+default keys, single-block decoupled lookback,
//  ScatterKeysGlobalDirect<false>/ScatterValuesGlobalDirect<false> with the
//  idx < tile_items guard. This is the path where the lifted sa-cuda binary
//  faults (key scatter at Onesweep+21196 with a small negative index).
//
// Build (cuda-13.2 toolkit — bench was built with CUDA 13.x cub):
//   /usr/local/cuda-13.2/bin/nvcc -arch=sm_90 -cubin partial.cu -o partial.cubin
//   /usr/local/cuda-13.2/bin/cuobjdump -sass partial.cubin > partial.sass

#include <cub/device/dispatch/kernels/kernel_radix_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_radix_sort.cuh>

using policy_hub_t = cub::detail::radix::policy_hub<int, int, unsigned long long>;

template __global__ void cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<
  policy_hub_t::Policy1000,
  cub::SortOrder::Ascending,
  int,                 // KeyT
  int,                 // ValueT
  unsigned long long,  // OffsetT
  int,                 // PortionOffsetT
  int,                 // AtomicOffsetT
  cub::detail::identity_decomposer_t>(
  int* d_lookback,
  int* d_ctrs,
  unsigned long long* d_bins_out,
  const unsigned long long* d_bins_in,
  int* d_keys_out,
  const int* d_keys_in,
  int* d_values_out,
  const int* d_values_in,
  int num_items,
  int current_bit,
  int num_bits,
  cub::detail::identity_decomposer_t decomposer);
