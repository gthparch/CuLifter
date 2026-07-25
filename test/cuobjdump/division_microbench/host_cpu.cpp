#include <cstdint>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void dm_div();
extern "C" void dm_mod();
extern "C" void dm_divmod();
extern "C" void dm_two_divisors();
extern "C" void dm_warpid();
extern "C" void dm_scalar();

static bool check_div(int divisor) {
    constexpr int BD = 32;
    int out[BD] = {0};
    launchKernel(dm_div, 1, BD, divisor, out);
    for (int t = 0; t < BD; ++t) {
        int expected = t / divisor;
        if (out[t] != expected) {
            std::cout << "dm_div FAIL t=" << t << " d=" << divisor
                      << " got=" << out[t] << " expected=" << expected << "\n";
            return false;
        }
    }
    return true;
}

static bool check_mod(int divisor) {
    constexpr int BD = 32;
    int out[BD] = {0};
    launchKernel(dm_mod, 1, BD, divisor, out);
    for (int t = 0; t < BD; ++t) {
        int expected = t % divisor;
        if (out[t] != expected) {
            std::cout << "dm_mod FAIL t=" << t << " d=" << divisor
                      << " got=" << out[t] << " expected=" << expected << "\n";
            return false;
        }
    }
    return true;
}

static bool check_divmod(int divisor) {
    constexpr int BD = 32;
    int q[BD] = {0}, r[BD] = {0};
    launchKernel(dm_divmod, 1, BD, divisor, q, r);
    for (int t = 0; t < BD; ++t) {
        if (q[t] != t / divisor || r[t] != t % divisor) {
            std::cout << "dm_divmod FAIL t=" << t << " d=" << divisor
                      << " q=" << q[t] << " r=" << r[t]
                      << " expected q=" << (t/divisor) << " r=" << (t%divisor) << "\n";
            return false;
        }
    }
    return true;
}

static bool check_two_divisors(int d1, int d2) {
    constexpr int BD = 32;
    int o1[BD] = {0}, o2[BD] = {0};
    launchKernel(dm_two_divisors, 1, BD, d1, d2, o1, o2);
    for (int t = 0; t < BD; ++t) {
        if (o1[t] != t / d1 || o2[t] != t / d2) {
            std::cout << "dm_two_divisors FAIL t=" << t
                      << " got=(" << o1[t] << "," << o2[t]
                      << ") expected=(" << (t/d1) << "," << (t/d2) << ")\n";
            return false;
        }
    }
    return true;
}

static bool check_scalar(int dividend, int divisor) {
    int out[2] = {0, 0};
    launchKernel(dm_scalar, 1, 32, dividend, divisor, out);
    int q = dividend / divisor, r = dividend % divisor;
    if (out[0] != q || out[1] != r) {
        std::cout << "dm_scalar FAIL dividend=" << dividend << " divisor=" << divisor
                  << " got=(" << out[0] << "," << out[1]
                  << ") expected=(" << q << "," << r << ")\n";
        return false;
    }
    return true;
}

int main() {
    bool ok = true;
    for (int d : {3, 7, 13, 25}) {
        ok &= check_div(d);
        ok &= check_mod(d);
        ok &= check_divmod(d);
    }
    ok &= check_two_divisors(3, 7);
    ok &= check_two_divisors(5, 11);
    ok &= check_scalar(1000, 7);
    ok &= check_scalar(123456, 25);

    if (ok) std::cout << "TEST PASSED\n";
    return ok ? 0 : 1;
}
