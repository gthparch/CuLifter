#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../common_ml.h"
#include "cuda_driver_helpers.h"

namespace {
constexpr char kKernelName[] = "ml_tensorop_gemm_bias_relu_f16";
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

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <ptx-or-cubin>\n";
    return 1;
  }

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

  CuDriverSession session;
  CUfunction fn = session.loadKernel(argv[1], kKernelName);

  CUdeviceptr d_a{};
  CUdeviceptr d_b{};
  CUdeviceptr d_bias{};
  CUdeviceptr d_c{};

  CUCHK(cuMemAlloc(&d_a, sizeof(uint16_t) * a.size()));
  CUCHK(cuMemAlloc(&d_b, sizeof(uint16_t) * b.size()));
  CUCHK(cuMemAlloc(&d_bias, sizeof(float) * bias.size()));
  CUCHK(cuMemAlloc(&d_c, sizeof(float) * c.size()));

  CUCHK(cuMemcpyHtoD(d_a, a.data(), sizeof(uint16_t) * a.size()));
  CUCHK(cuMemcpyHtoD(d_b, b.data(), sizeof(uint16_t) * b.size()));
  CUCHK(cuMemcpyHtoD(d_bias, bias.data(), sizeof(float) * bias.size()));
  CUCHK(cuMemcpyHtoD(d_c, c.data(), sizeof(float) * c.size()));

  void* params[] = {&d_a, &d_b, &d_bias, &d_c};
  CUCHK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr));
  CUCHK(cuCtxSynchronize());

  CUCHK(cuMemcpyDtoH(c.data(), d_c, sizeof(float) * c.size()));

  float err = ml_tensorop::max_abs_diff(c, ref);

  CUCHK(cuMemFree(d_a));
  CUCHK(cuMemFree(d_b));
  CUCHK(cuMemFree(d_bias));
  CUCHK(cuMemFree(d_c));

  if (err > 3e-2f) {
    std::cerr << "FAIL max_abs_diff=" << err << "\n";
    return 1;
  }

  std::cout << "TEST PASSED\n";
  return 0;
}
