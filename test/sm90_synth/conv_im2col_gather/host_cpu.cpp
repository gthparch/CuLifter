// x86 host driver for the CUTLASS group-conv im2col ACTIVATION GATHER audit
// (42_ampere_tensorop_group_conv, analytic activation iterator).
//
// Reads the oracle-generated geometry (config.txt), the exact-f16 activation
// tensor (activation.u16), and the expected compacted im2col matrix
// (gather_oracle.f32), launches the lifted kernel, and compares:
//   - WRITE-SET: Activation read-only (untouched); GatherOut fully written
//                (no still-canary, no stray beyond footprint).
//   - VALUES: every GatherOut[row*CRS+step] bit-exact vs oracle.
//
// Param offsets (sm90, base 0x210), dense 4B packing — verified vs SASS
// c[0x0][...] reads:
//   0x210 Activation(half*)  0x218 GatherOut(float*)
//   0x220 N 0x224 H 0x228 W 0x22c C 0x230 P 0x234 Q 0x238 R 0x23c S
//   0x240 pad_h 0x244 pad_w 0x248 stride_h 0x24c stride_w
//   0x250 dil_h 0x254 dil_w 0x258 groups 0x25c kColumn 0x260 CRS
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include "kernel_wrapper.h"

extern "C" void kern_im2col_gather();

static std::map<std::string,long> read_config(const char* path) {
    std::map<std::string,long> m;
    FILE* f = fopen(path, "r");
    if (!f) { printf("FAILED: cannot open %s\n", path); exit(2); }
    char key[64]; long val;
    while (fscanf(f, "%63s %ld", key, &val) == 2) m[key] = val;
    fclose(f);
    return m;
}
template<class T>
static std::vector<T> read_bin(const char* path, size_t count) {
    std::vector<T> v(count);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("FAILED: cannot open %s\n", path); exit(2); }
    size_t got = fread(v.data(), sizeof(T), count, f);
    fclose(f);
    if (got != count) { printf("FAILED: short read %s (%zu/%zu)\n", path, got, count); exit(2); }
    return v;
}

int main() {
    auto cfg = read_config("config.txt");
    const int N=cfg["N"], H=cfg["H"], W=cfg["W"], C=cfg["C"];
    const int P=cfg["P"], Q=cfg["Q"], R=cfg["R"], S=cfg["S"];
    const int pad_h=cfg["pad_h"], pad_w=cfg["pad_w"];
    const int stride_h=cfg["stride_h"], stride_w=cfg["stride_w"];
    const int dil_h=cfg["dil_h"], dil_w=cfg["dil_w"];
    const int groups=cfg["groups"], kColumn=cfg["kColumn"];
    const int CRS=cfg["CRS"];
    const long NPQ=cfg["NPQ"], nelem=cfg["nelem"];
    const long total = NPQ * (long)CRS;

    // Activation: raw u16 half bits -> store into a uint16 buffer (the lifted
    // kernel does LDG.U16 then HADD2.F32 / __half2float). The wrapper sees the
    // pointer as bytes; the kernel reads 2-byte halfs.
    auto act_u16 = read_bin<uint16_t>("activation.u16", (size_t)nelem);
    auto oracle  = read_bin<float>("gather_oracle.f32", (size_t)total);

    // GatherOut canary: distinct sentinel to detect unwritten / stray writes.
    const float CANARY = -123456.0f;
    std::vector<float> out((size_t)total, CANARY);
    // Guard tail: extra cells after the footprint to catch overruns.
    const size_t GUARD = 64;
    std::vector<float> out_buf((size_t)total + GUARD, CANARY);

    // Keep a copy of activation to confirm read-only.
    std::vector<uint16_t> act_copy = act_u16;

    const int total_threads = (int)total;
    const int block = 128;
    const int grid = (total_threads + block - 1) / block;

    launchKernel(kern_im2col_gather, grid, block,
        ArgAt{0x210}, act_u16.data(),
        ArgAt{0x218}, out_buf.data(),
        ArgAt{0x220}, N,
        ArgAt{0x224}, H,
        ArgAt{0x228}, W,
        ArgAt{0x22c}, C,
        ArgAt{0x230}, P,
        ArgAt{0x234}, Q,
        ArgAt{0x238}, R,
        ArgAt{0x23c}, S,
        ArgAt{0x240}, pad_h,
        ArgAt{0x244}, pad_w,
        ArgAt{0x248}, stride_h,
        ArgAt{0x24c}, stride_w,
        ArgAt{0x250}, dil_h,
        ArgAt{0x254}, dil_w,
        ArgAt{0x258}, groups,
        ArgAt{0x25c}, kColumn,
        ArgAt{0x260}, CRS);

    // --- WRITE-SET checks ---
    int act_dirty = 0;
    for (size_t i = 0; i < (size_t)nelem; i++)
        if (act_u16[i] != act_copy[i]) act_dirty++;

    int still_canary = 0;
    for (long i = 0; i < total; i++)
        if (out_buf[i] == CANARY) still_canary++;
    int guard_stray = 0;
    for (size_t i = (size_t)total; i < (size_t)total + GUARD; i++)
        if (out_buf[i] != CANARY) guard_stray++;

    // --- VALUE checks (bit-exact) ---
    long mism = 0;
    double max_abs = 0.0, max_rel = 0.0;
    long first_bad = -1;
    for (long i = 0; i < total; i++) {
        float got = out_buf[i], exp = oracle[i];
        if (memcmp(&got, &exp, 4) != 0) {
            if (first_bad < 0) first_bad = i;
            mism++;
            double a = std::fabs((double)got - (double)exp);
            if (a > max_abs) max_abs = a;
            double denom = std::fabs((double)exp);
            double r = denom > 0 ? a/denom : a;
            if (r > max_rel) max_rel = r;
        }
    }

    printf("=== conv_im2col_gather (42 group-conv analytic activation iterator) ===\n");
    printf("geom: N=%d H=%d W=%d C=%d P=%d Q=%d R=%d S=%d pad=(%d,%d) groups=%d kColumn=%d\n",
           N,H,W,C,P,Q,R,S,pad_h,pad_w,groups,kColumn);
    printf("P*Q=%d (pow2? %s)  CRS=%d  total cells=%ld\n",
           P*Q, ((P*Q)&(P*Q-1))==0 ? "yes":"NO (runtime divide)", CRS, total);
    printf("WRITESET: activation dirty=%d  still-canary=%d  guard-stray=%d\n",
           act_dirty, still_canary, guard_stray);
    printf("VALUES:   mismatches=%ld/%ld  max_abs=%g  max_rel=%g  first_bad=%ld\n",
           mism, total, max_abs, max_rel, first_bad);

    if (act_dirty==0 && still_canary==0 && guard_stray==0 && mism==0) {
        printf("REF VERIFIED (im2col gather bit-exact: write-set complete + values exact)\n");
        printf("TEST PASSED\n");
        return 0;
    }
    if (first_bad >= 0) {
        long row = first_bad / CRS, step = first_bad % CRS;
        printf("FIRST BAD: cell %ld (npq_row=%ld crs_step=%ld) got=%g exp=%g\n",
               first_bad, row, step, out_buf[first_bad], oracle[first_bad]);
    }
    printf("FAILED\n");
    return 1;
}
