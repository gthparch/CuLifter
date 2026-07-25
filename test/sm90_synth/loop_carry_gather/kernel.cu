// MINIMAL reduction of the 42_group_conv im2col divergence: a per-thread loop
// that REPLAYS an advance() cascade (loop-carried induction vars with a
// data-dependent `continue` / nested counter rollover), then uses the FINAL
// induction state to index a gather. Mirrors the conv iterator's filter_(r,s,c)
// advance + gather, stripped of all conv geometry.
//
// For thread t (= steps to replay), walk:  cnt++; s++; if(s<S)continue;
//   s=0; r++; if(r<R)continue; r=0; if(cnt==G){cnt=0;grp++;c=grp*CPG;} else c+=K;
// then  idx = ((r*S + s) + c) % SRC  ;  out[t] = src[idx].
//
// Expected: out[t] varies with t exactly as the replayed (r,s,c) dictate.
// Bug symptom (from the conv microbench): out[] comes out CONSTANT per group /
// stale, because the loop-carried induction is not threaded to the gather.
extern "C" __global__ void kern_loop_carry_gather(
    const float* __restrict__ src,
    float* __restrict__ out,
    int T, int S, int R, int G, int CPG, int K, int SRC)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    int r = 0, s = 0, c = 0, cnt = 0, grp = 0;
    for (int step = 0; step < t; ++step) {
        ++cnt;
        ++s;
        if (s < S) continue;
        s = 0;
        ++r;
        if (r < R) continue;
        r = 0;
        if (cnt == G) { cnt = 0; ++grp; c = grp * CPG; }
        else          { c += K; }
    }
    int idx = ((r * S + s) + c) % SRC;
    out[t] = src[idx];
}
