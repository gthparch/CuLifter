// Driver for division_microbench.cu: runs each int-division kernel and
// compares against a CPU reference. Designed to bisect the SLifter NVVM
// lifter's mishandling of ptxas's runtime unsigned-divide-by-reciprocal
// pattern (Hacker's-Delight magic multiplication).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

bool kernel_dm_div(CuDriverSession& session, const char* img, std::ostream& err) {
    CUfunction fn = session.loadKernel(img, "dm_div");
    int d = 7;  // not power-of-2 so ptxas uses reciprocal-mul, not shift
    std::vector<int> out(64, 0);
    CUdeviceptr d_out{};
    CUCHK(cuMemAlloc(&d_out, out.size() * sizeof(int)));
    CUCHK(cuMemsetD8(d_out, 0, out.size() * sizeof(int)));
    void* params[] = {&d, &d_out};
    CUCHK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr));
    CUCHK(cuCtxSynchronize());
    CUCHK(cuMemcpyDtoH(out.data(), d_out, out.size() * sizeof(int)));
    CUCHK(cuMemFree(d_out));
    for (int t = 0; t < 32; ++t) {
        if (out[t] != t / d) {
            err << "FAIL dm_div at t=" << t << ": got " << out[t]
                << ", expected " << (t / d) << " (d=" << d << ")\n";
            return false;
        }
    }
    return true;
}

bool kernel_dm_mod(CuDriverSession& session, const char* img, std::ostream& err) {
    CUfunction fn = session.loadKernel(img, "dm_mod");
    int d = 7;
    std::vector<int> out(64, 0);
    CUdeviceptr d_out{};
    CUCHK(cuMemAlloc(&d_out, out.size() * sizeof(int)));
    CUCHK(cuMemsetD8(d_out, 0, out.size() * sizeof(int)));
    void* params[] = {&d, &d_out};
    CUCHK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr));
    CUCHK(cuCtxSynchronize());
    CUCHK(cuMemcpyDtoH(out.data(), d_out, out.size() * sizeof(int)));
    CUCHK(cuMemFree(d_out));
    for (int t = 0; t < 32; ++t) {
        if (out[t] != t % d) {
            err << "FAIL dm_mod at t=" << t << ": got " << out[t]
                << ", expected " << (t % d) << " (d=" << d << ")\n";
            return false;
        }
    }
    return true;
}

bool kernel_dm_divmod(CuDriverSession& session, const char* img, std::ostream& err) {
    CUfunction fn = session.loadKernel(img, "dm_divmod");
    int d = 7;
    std::vector<int> q_out(64, 0), r_out(64, 0);
    CUdeviceptr d_q{}, d_r{};
    CUCHK(cuMemAlloc(&d_q, q_out.size() * sizeof(int)));
    CUCHK(cuMemAlloc(&d_r, r_out.size() * sizeof(int)));
    CUCHK(cuMemsetD8(d_q, 0, q_out.size() * sizeof(int)));
    CUCHK(cuMemsetD8(d_r, 0, r_out.size() * sizeof(int)));
    void* params[] = {&d, &d_q, &d_r};
    CUCHK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr));
    CUCHK(cuCtxSynchronize());
    CUCHK(cuMemcpyDtoH(q_out.data(), d_q, q_out.size() * sizeof(int)));
    CUCHK(cuMemcpyDtoH(r_out.data(), d_r, r_out.size() * sizeof(int)));
    CUCHK(cuMemFree(d_q));
    CUCHK(cuMemFree(d_r));
    for (int t = 0; t < 32; ++t) {
        if (q_out[t] != t / d) {
            err << "FAIL dm_divmod (q) at t=" << t << ": got " << q_out[t]
                << ", expected " << (t / d) << "\n";
            return false;
        }
        if (r_out[t] != t % d) {
            err << "FAIL dm_divmod (r) at t=" << t << ": got " << r_out[t]
                << ", expected " << (t % d) << "\n";
            return false;
        }
    }
    return true;
}

bool kernel_dm_two_divisors(CuDriverSession& session, const char* img, std::ostream& err) {
    CUfunction fn = session.loadKernel(img, "dm_two_divisors");
    int d1 = 7, d2 = 11;
    std::vector<int> out1(64, 0), out2(64, 0);
    CUdeviceptr d_o1{}, d_o2{};
    CUCHK(cuMemAlloc(&d_o1, out1.size() * sizeof(int)));
    CUCHK(cuMemAlloc(&d_o2, out2.size() * sizeof(int)));
    CUCHK(cuMemsetD8(d_o1, 0, out1.size() * sizeof(int)));
    CUCHK(cuMemsetD8(d_o2, 0, out2.size() * sizeof(int)));
    void* params[] = {&d1, &d2, &d_o1, &d_o2};
    CUCHK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr));
    CUCHK(cuCtxSynchronize());
    CUCHK(cuMemcpyDtoH(out1.data(), d_o1, out1.size() * sizeof(int)));
    CUCHK(cuMemcpyDtoH(out2.data(), d_o2, out2.size() * sizeof(int)));
    CUCHK(cuMemFree(d_o1));
    CUCHK(cuMemFree(d_o2));
    for (int t = 0; t < 32; ++t) {
        if (out1[t] != t / d1 || out2[t] != t / d2) {
            err << "FAIL dm_two_divisors at t=" << t
                << ": got (" << out1[t] << ", " << out2[t]
                << "), expected (" << (t / d1) << ", " << (t / d2) << ")\n";
            return false;
        }
    }
    return true;
}

bool kernel_dm_warpid(CuDriverSession& session, const char* img, std::ostream& err) {
    CUfunction fn = session.loadKernel(img, "dm_warpid");
    int N = 64;
    int total_warps = 16;
    std::vector<int> out(total_warps * 2, 0);
    CUdeviceptr d_out{};
    CUCHK(cuMemAlloc(&d_out, out.size() * sizeof(int)));
    CUCHK(cuMemsetD8(d_out, 0, out.size() * sizeof(int)));
    void* params[] = {&N, &d_out};
    CUCHK(cuLaunchKernel(fn, 4, 1, 1, 128, 1, 1, 0, nullptr, params, nullptr));
    CUCHK(cuCtxSynchronize());
    CUCHK(cuMemcpyDtoH(out.data(), d_out, out.size() * sizeof(int)));
    CUCHK(cuMemFree(d_out));
    int warpN = (N + 15) / 16;
    for (int w = 0; w < total_warps; ++w) {
        int got_q = out[w * 2 + 0], got_r = out[w * 2 + 1];
        int exp_q = w / warpN,      exp_r = w % warpN;
        if (got_q != exp_q || got_r != exp_r) {
            err << "FAIL dm_warpid at warp=" << w
                << ": got (q=" << got_q << ", r=" << got_r
                << "), expected (q=" << exp_q << ", r=" << exp_r
                << ") (warpN=" << warpN << ")\n";
            return false;
        }
    }
    return true;
}

bool kernel_dm_scalar(CuDriverSession& session, const char* img, std::ostream& err) {
    CUfunction fn = session.loadKernel(img, "dm_scalar");
    int dividend = 100, divisor = 7;
    std::vector<int> out(2, -1);
    CUdeviceptr d_out{};
    CUCHK(cuMemAlloc(&d_out, out.size() * sizeof(int)));
    CUCHK(cuMemsetD8(d_out, 0, out.size() * sizeof(int)));
    void* params[] = {&dividend, &divisor, &d_out};
    CUCHK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr));
    CUCHK(cuCtxSynchronize());
    CUCHK(cuMemcpyDtoH(out.data(), d_out, out.size() * sizeof(int)));
    CUCHK(cuMemFree(d_out));
    int exp_q = dividend / divisor, exp_r = dividend % divisor;
    if (out[0] != exp_q || out[1] != exp_r) {
        err << "FAIL dm_scalar: got (q=" << out[0] << ", r=" << out[1]
            << "), expected (q=" << exp_q << ", r=" << exp_r << ")\n";
        return false;
    }
    return true;
}

struct Case {
    const char* name;
    bool (*run)(CuDriverSession&, const char*, std::ostream&);
};

const Case kCases[] = {
    {"dm_div",          kernel_dm_div},
    {"dm_mod",          kernel_dm_mod},
    {"dm_divmod",       kernel_dm_divmod},
    {"dm_two_divisors", kernel_dm_two_divisors},
    {"dm_scalar",       kernel_dm_scalar},
    {"dm_warpid",       kernel_dm_warpid},
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <cubin|ptx> [kernel_name_filter]\n";
        return 1;
    }
    const char* device_img = argv[1];
    const char* filter = (argc >= 3) ? argv[2] : nullptr;

    CuDriverSession session;
    bool all_ok = true;
    int run = 0, passed = 0;
    for (const auto& tc : kCases) {
        if (filter && std::strcmp(filter, tc.name) != 0) continue;
        ++run;
        if (tc.run(session, device_img, std::cerr)) {
            ++passed;
        } else {
            all_ok = false;
            break;
        }
    }
    if (run == 0) {
        std::cerr << "no matching kernel for filter " << (filter ? filter : "<none>") << "\n";
        return 1;
    }
    if (all_ok) {
        std::cout << "TEST PASSED (" << passed << "/" << run << " kernels)\n";
        return 0;
    }
    return 1;
}
