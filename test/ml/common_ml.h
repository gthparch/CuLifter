#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace ml_tensorop {

inline uint32_t bit_cast_u32(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}

inline float bit_cast_f32(uint32_t x) {
  float v;
  std::memcpy(&v, &x, sizeof(v));
  return v;
}

inline uint16_t float_to_half_bits(float x) {
  uint32_t bits = bit_cast_u32(x);
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;

  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(1 - exp);
    uint32_t half_mant = mant >> (shift + 13);
    uint32_t round_bit = (mant >> (shift + 12)) & 1u;
    return static_cast<uint16_t>(sign | (half_mant + round_bit));
  }

  if (exp >= 31) {
    if (mant != 0) {
      return static_cast<uint16_t>(sign | 0x7e00u);
    }
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  uint32_t half_exp = static_cast<uint32_t>(exp) << 10;
  uint32_t half_mant = mant >> 13;
  uint32_t round_bit = (mant >> 12) & 1u;
  return static_cast<uint16_t>(sign | half_exp | (half_mant + round_bit));
}

inline float half_bits_to_float(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;

  if (exp == 0) {
    if (mant == 0) {
      return bit_cast_f32(sign);
    }
    float v = std::ldexp(static_cast<float>(mant), -24);
    return (sign ? -v : v);
  }

  if (exp == 31) {
    if (mant == 0) {
      return bit_cast_f32(sign | 0x7f800000u);
    }
    return std::numeric_limits<float>::quiet_NaN();
  }

  uint32_t fexp = (exp + 112u) << 23;
  uint32_t fmant = mant << 13;
  return bit_cast_f32(sign | fexp | fmant);
}

inline void fill_random_half(std::vector<uint16_t>& out, int seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (uint16_t& x : out) {
    x = float_to_half_bits(dist(rng));
  }
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float out = 0.0f;
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    out = std::max(out, std::fabs(a[i] - b[i]));
  }
  return out;
}

}  // namespace ml_tensorop
