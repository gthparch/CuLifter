// Algorithm-specific x86 oracle for sortKV-cuda
// (thrust::sort_by_key -> cub radix sort).
//
// The benchmark's auto-generated one-shot driver cannot test this app: the
// Onesweep path is a multi-launch algorithm whose agent state (lookback tiles,
// bin counters) is wired between launches by the cub host dispatcher. But cub
// dispatches the ENTIRE sort to DeviceRadixSortSingleTileKernel (one launch,
// one block, no inter-launch state) when n <= the single-tile capacity — so
// driving that kernel with a shuffled permutation and a strict value check is
// a faithful replication of the vanilla launch sequence for small n, and a
// non-vacuous full-sort correctness test.
//
// Mirrors main.cu's verify loop: after sorting shuffled keys 0..n-1 with
// vals[i] = keys[i] % 256, expect keys_out[i] == i and vals_out[i] == i % 256.
//
// ELF truth (KPARAM_INFO, sm_90 cubin): params at c[0][0x210]:
//   0x210 const KeyT* keys_in   0x218 KeyT* keys_out
//   0x220 const ValT* vals_in   0x228 ValT* vals_out
//   0x230 u64 num_items         0x238 int current_bit
//   0x23c int end_bit           0x240 identity_decomposer_t (1 byte, unread)
// Launch bounds (EIATTR_MAX_THREADS) = 256 -> blockDim must be 256.
// writeArg natural alignment reproduces this layout exactly (no ArgAt pins).
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>
#include "kernel_wrapper.h"

// cub::CUB_300104_SM_900::detail::radix_sort::DeviceRadixSortSingleTileKernel
// <policy_hub<int, T, u64>::Policy1000, SortOrder::Ascending, int, T, u64,
//  identity_decomposer_t> for T in {long, short, int, unsigned char}.
extern "C" void _ZN3cub17CUB_300104_SM_9006detail10radix_sort31DeviceRadixSortSingleTileKernelINS1_5radix10policy_hubIilyE10Policy1000ELNS0_9SortOrderE0EilyNS1_21identity_decomposer_tEEEvPKT1_PSA_PKT2_PSE_T3_iiT4_();
extern "C" void _ZN3cub17CUB_300104_SM_9006detail10radix_sort31DeviceRadixSortSingleTileKernelINS1_5radix10policy_hubIisyE10Policy1000ELNS0_9SortOrderE0EisyNS1_21identity_decomposer_tEEEvPKT1_PSA_PKT2_PSE_T3_iiT4_();
extern "C" void _ZN3cub17CUB_300104_SM_9006detail10radix_sort31DeviceRadixSortSingleTileKernelINS1_5radix10policy_hubIiiyE10Policy1000ELNS0_9SortOrderE0EiiyNS1_21identity_decomposer_tEEEvPKT1_PSA_PKT2_PSE_T3_iiT4_();
extern "C" void _ZN3cub17CUB_300104_SM_9006detail10radix_sort31DeviceRadixSortSingleTileKernelINS1_5radix10policy_hubIihyE10Policy1000ELNS0_9SortOrderE0EihyNS1_21identity_decomposer_tEEEvPKT1_PSA_PKT2_PSE_T3_iiT4_();

namespace {

constexpr int N = 1024;            // <= single-tile capacity (Policy1000, 256 thr)
constexpr int BLOCK_THREADS = 256; // ELF launch bounds

uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }

template <typename ValT, typename Fn>
int run_one(const char* tag, Fn fn) {
    int*  keys_in  = new int[N];
    int*  keys_out = new int[N]();
    ValT* vals_in  = new ValT[N];
    ValT* vals_out = new ValT[N]();

    // Deterministic shuffled permutation of 0..N-1 (Fisher-Yates, LCG seed 123)
    for (int i = 0; i < N; i++) keys_in[i] = i;
    uint32_t seed = 123;
    for (int i = N - 1; i > 0; i--) {
        int j = (int)(lcg(seed) % (uint32_t)(i + 1));
        std::swap(keys_in[i], keys_in[j]);
    }
    for (int i = 0; i < N; i++) vals_in[i] = (ValT)(keys_in[i] % 256);

    launchKernel(fn, /*gridDim=*/1, /*blockDim=*/BLOCK_THREADS,
                 (const int*)keys_in, keys_out,
                 (const ValT*)vals_in, vals_out,
                 (unsigned long long)N,
                 (int)0, (int)32,           // current_bit, end_bit (thrust defaults)
                 (unsigned char)0);         // identity_decomposer_t (1B, unread)

    int bad = 0, first = -1;
    for (int i = 0; i < N; i++) {
        if (keys_out[i] != i || vals_out[i] != (ValT)(i % 256)) {
            bad++;
            if (first < 0) first = i;
        }
    }
    if (bad) {
        std::cout << "FAIL: " << tag << " " << bad << "/" << N
                  << " mismatches, first at i=" << first
                  << " key=" << keys_out[first]
                  << " val=" << (long long)vals_out[first] << std::endl;
        return 1;
    }
    std::cout << tag << ": " << N << " pairs sorted correctly" << std::endl;
    return 0;
}

} // namespace

int main() {
    int rc = 0;
    rc |= run_one<unsigned char>("KV<int,uchar>", _ZN3cub17CUB_300104_SM_9006detail10radix_sort31DeviceRadixSortSingleTileKernelINS1_5radix10policy_hubIihyE10Policy1000ELNS0_9SortOrderE0EihyNS1_21identity_decomposer_tEEEvPKT1_PSA_PKT2_PSE_T3_iiT4_);
    rc |= run_one<short>("KV<int,short>", _ZN3cub17CUB_300104_SM_9006detail10radix_sort31DeviceRadixSortSingleTileKernelINS1_5radix10policy_hubIisyE10Policy1000ELNS0_9SortOrderE0EisyNS1_21identity_decomposer_tEEEvPKT1_PSA_PKT2_PSE_T3_iiT4_);
    rc |= run_one<int>("KV<int,int>", _ZN3cub17CUB_300104_SM_9006detail10radix_sort31DeviceRadixSortSingleTileKernelINS1_5radix10policy_hubIiiyE10Policy1000ELNS0_9SortOrderE0EiiyNS1_21identity_decomposer_tEEEvPKT1_PSA_PKT2_PSE_T3_iiT4_);
    rc |= run_one<long long>("KV<int,long>", _ZN3cub17CUB_300104_SM_9006detail10radix_sort31DeviceRadixSortSingleTileKernelINS1_5radix10policy_hubIilyE10Policy1000ELNS0_9SortOrderE0EilyNS1_21identity_decomposer_tEEEvPKT1_PSA_PKT2_PSE_T3_iiT4_);
    if (rc) return 1;
    std::cout << "TEST PASSED" << std::endl;
    return 0;
}
