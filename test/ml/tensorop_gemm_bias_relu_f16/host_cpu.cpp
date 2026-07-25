#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../common_ml.h"
#include "kernel_wrapper.h"

extern "C" void ml_tensorop_gemm_bias_relu_f16();

namespace {
constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;
constexpr int kElems = kM * kN;

void reference_gemm_bias_relu(const std::vector<uint16_t>& a,
                              const std::vector<uint16_t>& b,
                              const std::vector<float>& bias,
                              std::vector<float>& out) {
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < kK; ++k) {
        float av = ml_tensorop::half_bits_to_float(a[m * kK + k]);
        float bv = ml_tensorop::half_bits_to_float(b[n * kK + k]);
        acc += av * bv;
      }
      float v = acc + bias[n];
      out[m * kN + n] = (v > 0.0f ? v : 0.0f);
    }
  }
}

}  // namespace

int main() {
  std::vector<uint16_t> a(kElems);
  std::vector<uint16_t> b(kElems);
  std::vector<float> bias(kN, 0.0f);
  std::vector<float> c(kElems, 0.0f);
  std::vector<float> ref(kElems, 0.0f);

  ml_tensorop::fill_random_half(a, 505, -1.0f, 1.0f);
  ml_tensorop::fill_random_half(b, 606, -1.0f, 1.0f);

  for (int i = 0; i < kN; ++i) {
    float f = static_cast<float>(i - 8) * 0.15f;
    bias[i] = f;
  }

  reference_gemm_bias_relu(a, b, bias, ref);

  launchKernel(ml_tensorop_gemm_bias_relu_f16, 1, 32, a.data(), b.data(), bias.data(), c.data());

  float err = ml_tensorop::max_abs_diff(c, ref);
  if (err > 3e-2f) {
    std::cerr << "FAIL max_abs_diff=" << err << "\n";
    return 1;
  }

  std::cout << "TEST PASSED\n";
  return 0;
}
