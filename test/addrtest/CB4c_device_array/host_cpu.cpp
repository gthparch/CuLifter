#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void CB4c_device_array();

// Access const_mem to populate c[4] with __device__ variable pointers
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];

int main() {
    constexpr int N = 256;
    int *h_in = new int[N];
    int *h_out = new int[N];
    for (int i = 0; i < N; i++) h_in[i] = i + 1;
    memset(h_out, 0, N * sizeof(int));

    // __device__ int d_arr[256];
    // SASS: LDC.64 R4, c[0x4][RZ]  -> load pointer from c[4][0..7]
    //       IMAD.WIDE R4, R9, 0x4, R4  -> &d_arr[i]
    //       STG.E desc[UR4][R4.64], R11  -> store to d_arr[i]
    // c[4][0..7] = pointer to d_arr array
    int *d_arr_ptr = new int[N];
    memset(d_arr_ptr, 0, N * sizeof(int));
    uint64_t addr = (uint64_t)(uintptr_t)d_arr_ptr;
    memcpy(&const_mem[4][0], &addr, sizeof(uint64_t));

    launchKernel(CB4c_device_array, (N+255)/256, 256, h_in, h_out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        int expected = (i + 1) * 3;
        if (h_out[i] != expected) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " expected " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] d_arr_ptr;
    delete[] h_in;
    delete[] h_out;
    return ok ? 0 : 1;
}
