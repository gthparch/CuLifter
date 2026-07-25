// x86 driver + self-contained oracle for the loop-carried-IV gather reduction.
// Computes the expected out[t] by replaying the SAME advance cascade on the host
// (independent of the lifter), then compares vs the lifted kernel bit-exactly.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "kernel_wrapper.h"

extern "C" void kern_loop_carry_gather();

int main() {
    const int T = 200;          // threads (replay 0..199 steps)
    const int S = 3, R = 3, G = 9, CPG = 8, K = 8;
    const int SRC = 4096;

    std::vector<float> src(SRC);
    for (int i = 0; i < SRC; i++) src[i] = (float)(i + 1) * 0.5f;   // exact f32
    std::vector<float> out(T, -999.0f);

    const int block = 64, grid = (T + block - 1) / block;
    launchKernel(kern_loop_carry_gather, grid, block,
        ArgAt{0x210}, src.data(),
        ArgAt{0x218}, out.data(),
        ArgAt{0x220}, T,
        ArgAt{0x224}, S,
        ArgAt{0x228}, R,
        ArgAt{0x22c}, G,
        ArgAt{0x230}, CPG,
        ArgAt{0x234}, K,
        ArgAt{0x238}, SRC);

    // --- host oracle: replay the cascade exactly as the kernel ---
    std::vector<float> exp(T);
    for (int t = 0; t < T; t++) {
        int r=0,s=0,c=0,cnt=0,grp=0;
        for (int step=0; step<t; ++step) {
            ++cnt; ++s;
            if (s<S) continue;
            s=0; ++r;
            if (r<R) continue;
            r=0;
            if (cnt==G){cnt=0;++grp;c=grp*CPG;} else {c+=K;}
        }
        int idx = ((r*S + s) + c) % SRC;
        exp[t] = src[idx];
    }

    int mism=0, first=-1;
    for (int t=0;t<T;t++) if (memcmp(&out[t],&exp[t],4)){ if(first<0)first=t; mism++; }
    printf("=== loop_carry_gather (minimal reduction of 42 im2col loop-carried IV) ===\n");
    printf("T=%d S=%d R=%d G=%d CPG=%d K=%d\n", T,S,R,G,CPG,K);
    printf("mismatches=%d/%d  first_bad=%d\n", mism, T, first);
    if (first>=0) {
        // recompute the expected (r,s,c) at first to show the stale-state signature
        int t=first,r=0,s=0,c=0,cnt=0,grp=0;
        for(int st=0;st<t;++st){++cnt;++s;if(s<S)continue;s=0;++r;if(r<R)continue;r=0;if(cnt==G){cnt=0;++grp;c=grp*CPG;}else c+=K;}
        printf("first bad t=%d: got=%g exp=%g  (expected r=%d s=%d c=%d -> idx=%d)\n",
               first, out[first], exp[first], r,s,c, ((r*S+s)+c)%SRC);
        // print out[] vs exp[] head to see the constant-per-group pattern
        printf("got : "); for(int i=0;i<24;i++)printf("%g ",out[i]); printf("\n");
        printf("exp : "); for(int i=0;i<24;i++)printf("%g ",exp[i]); printf("\n");
    }
    if (mism==0) { printf("REF VERIFIED (loop-carried IV gather bit-exact)\nTEST PASSED\n"); return 0; }
    printf("FAILED\n");
    return 1;
}
