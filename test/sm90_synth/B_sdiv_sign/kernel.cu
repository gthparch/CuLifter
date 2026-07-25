// Microbench: signed 32-bit divide + modulo by a RUNTIME divisor.
// ptxas lowers this to the Granlund-Möller MUFU.RCP magic chain with
// IABS on both operands and a sign fixup; IntDivRecognize must splice
// the chain's unsigned magnitudes (NOT the signed originals) into SDIV,
// or every negative-operand quotient is wrong (idivide-cuda, H200 pilot
// 142532: 33664 / -1 -> -2).
extern "C" __global__ void kern_sdiv_sign(int* q, int* r, const int* a,
                                          const int* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    q[i] = a[i] / b[i];
    r[i] = a[i] % b[i];
}
