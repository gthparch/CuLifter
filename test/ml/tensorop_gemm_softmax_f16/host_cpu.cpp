#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../common_ml.h"
#include "kernel_wrapper.h"

extern "C" void ml_tensorop_gemm_softmax_f16();

namespace {
constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;
constexpr int kElems = kM * kN;

void reference_gemm_softmax(const std::vector<uint16_t>& a,
                            const std::vector<uint16_t>& b,
                            std::vector<float>& out) {
  std::vector<float> gemm(kElems, 0.0f);

  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < kK; ++k) {
        float av = ml_tensorop::half_bits_to_float(a[m * kK + k]);
        float bv = ml_tensorop::half_bits_to_float(b[n * kK + k]);
        acc += av * bv;
      }
      gemm[m * kN + n] = acc;
    }
  }

  for (int m = 0; m < kM; ++m) {
    float max_v = -1.0e20f;
    for (int n = 0; n < kN; ++n) {
      max_v = std::max(max_v, gemm[m * kN + n]);
    }

    float sum = 0.0f;
    for (int n = 0; n < kN; ++n) {
      sum += std::exp(gemm[m * kN + n] - max_v);
    }

    for (int n = 0; n < kN; ++n) {
      out[m * kN + n] = std::exp(gemm[m * kN + n] - max_v) / sum;
    }
  }
}

}  // namespace

int main() {
  std::vector<uint16_t> a(kElems);
  std::vector<uint16_t> b(kElems);
  std::vector<float> c(kElems, 0.0f);
  std::vector<float> ref(kElems, 0.0f);

  ml_tensorop::fill_random_half(a, 707, -1.0f, 1.0f);
  ml_tensorop::fill_random_half(b, 808, -1.0f, 1.0f);
  reference_gemm_softmax(a, b, ref);

  launchKernel(ml_tensorop_gemm_softmax_f16, 1, 32, a.data(), b.data(), c.data());

  float err = ml_tensorop::max_abs_diff(c, ref);
  if (err > 5e-3f) {
    std::cerr << "FAIL max_abs_diff=" << err << "\n";
    return 1;
  }

  std::cout << "TEST PASSED\n";
  return 0;
}
