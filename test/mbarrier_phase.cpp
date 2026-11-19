#include "common/kernel_wrapper_75.h"

int main() {
    constexpr uint32_t offset = 64;
    for (unsigned phase : {0u, 1u}) {
        const uint64_t state = (uint64_t{phase} << 63) | 0x1ffffc;
        slifter_mbar_init(offset, state);
        for (int arrivals = 0; arrivals <= 2; ++arrivals) {
            const unsigned current = phase ^ (arrivals == 2);
            for (unsigned parity : {0u, 1u}) {
                const int ready = slifter_mbar_try_wait(offset, parity << 31);
                if (ready != (current != parity)) {
                    std::cerr << "phase=" << phase << " arrivals=" << arrivals
                              << " parity=" << parity << " ready=" << ready << '\n';
                    return 1;
                }
            }
            if (arrivals < 2)
                slifter_mbar_arrive(offset, 0);
        }
    }
}
