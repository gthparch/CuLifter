#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../common_ml.h"
#include "kernel_wrapper.h"

extern "C" void ml_tensorop_batched_gemm_f16();

namespace {
constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;
constexpr int kBatchCount = 3;
constexpr int kTileElems = kM * kN;

void reference_batched_gemm(const std::vector<uint16_t>& a,
                            const std::vector<uint16_t>& b,
                            std::vector<float>& out) {
  for (int batch = 0; batch < kBatchCount; ++batch) {
    const int base = batch * kTileElems;
    for (int m = 0; m < kM; ++m) {
      for (int n = 0; n < kN; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < kK; ++k) {
          float av = ml_tensorop::half_bits_to_float(a[base + m * kK + k]);
          float bv = ml_tensorop::half_bits_to_float(b[base + n * kK + k]);
          acc += av * bv;
        }
        out[base + m * kN + n] = acc;
      }
    }
  }
}

}  // namespace

int main() {
  std::vector<uint16_t> a(kBatchCount * kTileElems);
  std::vector<uint16_t> b(kBatchCount * kTileElems);
  std::vector<float> c(kBatchCount * kTileElems, 0.0f);
  std::vector<float> ref(kBatchCount * kTileElems, 0.0f);

  ml_tensorop::fill_random_half(a, 303, -1.0f, 1.0f);
  ml_tensorop::fill_random_half(b, 404, -1.0f, 1.0f);
  reference_batched_gemm(a, b, ref);

  launchKernel(ml_tensorop_batched_gemm_f16, kBatchCount, 32, a.data(), b.data(), c.data());

  float err = ml_tensorop::max_abs_diff(c, ref);
  if (err > 3e-2f) {
    std::cerr << "FAIL max_abs_diff=" << err << "\n";
    return 1;
  }

  std::cout << "TEST PASSED\n";
  return 0;
}
