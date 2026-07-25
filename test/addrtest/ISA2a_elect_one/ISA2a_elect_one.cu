#include <cstdint>
extern "C" __global__
void ISA2a_elect_one(int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // elect.sync: predicate output + register output
        int is_elected;
        asm volatile(
            "{\n\t"
            "  .reg .pred p;\n\t"
            "  elect.sync _|p, 0xFFFFFFFF;\n\t"
            "  selp.s32 %0, 1, 0, p;\n\t"
            "}\n\t"
            : "=r"(is_elected)
        );
        if (is_elected) {
            out[0] = tid;
        }
    }
}
