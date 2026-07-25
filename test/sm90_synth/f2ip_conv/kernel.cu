// F2IP.S8.F32 fixture — the sm90 float->int8 pack-pair convert.
//
// F2IP is ptxas's fused lowering of PTX `cvt.pack.sat.s8.s32.b32 d, a, b, c`
// when the s32 inputs are themselves `cvt.rni.s32.f32` of floats — i.e. the
// CUTLASS NumericArrayConverter<int8_t, float, N> epilogue convert that quantizes
// fp32 accumulators to packed int8 before the global store. The int8 GEMM examples
// (13_fused_two_gemms_s8_sm80) emit 176 of these after the IMMA.16832 mainloop.
//
// Semantics of `F2IP.S8.F32.NTZ Rd, Ra, Rb, Rc` (decoded from PTX ISA + the CUTLASS
// 4-wide converter `cvt.pack r4,%4,%3,0; cvt.pack %0,%2,%1,r4`):
//   Rd[ 7: 0] = sat_s8( round_nearest_even( Rb ) )   (second SASS source -> byte0)
//   Rd[15: 8] = sat_s8( round_nearest_even( Ra ) )   (first  SASS source -> byte1)
//   Rd[31:16] = Rc[15:0]                              (accumulator low half -> bytes 2,3)
//   .NTZ = round-to-nearest-even + flush-denorm; .S8 = signed saturate to [-128,127].
//
// This kernel reproduces the exact CUTLASS Array<int8_t,4> <= Array<float,4> path:
// per element i it converts 4 floats (in[i*4 .. i*4+3]) to s8 and packs them into a
// single 32-bit word (byte j = s8(round(in[i*4+j]))), via TWO cvt.pack/F2IP calls,
// matching the benchmark's two-F2IP-per-32-bit-reg epilogue. Host oracle in host_cpu.cpp.
extern "C" __global__ void kern_f2ip(const float* in, int* out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Convert each float to a rounded s32 (cvt.rni.s32.f32), then pack-saturate to
    // s8 with cvt.pack.sat.s8 — ptxas fuses the rni+pack into F2IP.S8.F32.NTZ.
    int a0 = __float2int_rn(in[i * 4 + 0]);
    int a1 = __float2int_rn(in[i * 4 + 1]);
    int a2 = __float2int_rn(in[i * 4 + 2]);
    int a3 = __float2int_rn(in[i * 4 + 3]);

    unsigned packed;
    // r4   = { 0, 0, s8(a3), s8(a2) }   (byte0=a2, byte1=a3)
    // out  = { s8(a3), s8(a2), s8(a1), s8(a0) }  (byte0=a0, byte1=a1, hi16 from r4)
    asm volatile(
        "{ .reg .u32 r4;"
        "cvt.pack.sat.s8.s32.b32 r4, %4, %3, 0;"
        "cvt.pack.sat.s8.s32.b32 %0, %2, %1, r4;"
        "}"
        : "=r"(packed)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3));

    out[i] = static_cast<int>(packed);
}
