#pragma once
#include <cstdint>
#include <type_traits>
#include <bit>
#include <iostream>
#include <cstring>

extern "C" alignas(8) uint8_t const_mem[4096] = {0};

enum : int {
    SR_NTID_X     = 0x08,
    SR_GRID_DIM_X = 0x14,
    SR_CTAID_X    = 0x20,
    SR_TID_X      = 0x2C,
    SR_LANE_ID    = 0x38,
    ARG_BASE      = 0x140
};

static uint32_t off = ARG_BASE;

static constexpr uint32_t align_up(uint32_t x, uint32_t a) {
    return (x + (a - 1u)) & ~(a - 1u);
}

template<typename T>
inline void writeArg(T v)
{
    uint32_t aligned_off = align_up(off, static_cast<uint32_t>(sizeof(T)));

    if constexpr (std::is_pointer_v<T>) {
        uint64_t raw = reinterpret_cast<uint64_t>(v);
        std::memcpy(&const_mem[aligned_off], &raw, sizeof(raw));
        std::cout << " ptr=0x" << std::hex << raw << std::dec;
    } else if constexpr (sizeof(T) == 8) {
        uint64_t raw = std::bit_cast<uint64_t>(v);
        std::memcpy(&const_mem[aligned_off], &raw, sizeof(raw));
        std::cout << " u64=0x" << std::hex << raw << std::dec;
    } else if constexpr (sizeof(T) == 4) {
        uint32_t raw = std::bit_cast<uint32_t>(v);
        std::memcpy(&const_mem[aligned_off], &raw, sizeof(raw));
        std::cout << " u32=0x" << std::hex << raw << std::dec;
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Unexpected arg size");
    }

    std::cout << " Arg byte_off=" << aligned_off << ", type=" << typeid(T).name()
              << ", size=" << sizeof(T) << std::endl;

    off = aligned_off + sizeof(T);
}

// Explicit-offset pin (see kernel_wrapper_75.h).
struct ArgAt { uint32_t abs_off; };
inline void writeArg(ArgAt a) { off = a.abs_off; }

template<typename Func, typename... Args>
void launchKernel(Func func, int gridDim, int blockDim, Args... args)
{
    off = ARG_BASE;

    // std::memset(&const_mem[ARG_BASE], 0, 4096 - ARG_BASE);

    (writeArg(args), ...);

    for (int cta = 0; cta < gridDim; ++cta) {
        for (int tid = 0; tid < blockDim; ++tid) {
            *reinterpret_cast<uint32_t*>(&const_mem[SR_NTID_X])     = static_cast<uint32_t>(blockDim);
            *reinterpret_cast<uint32_t*>(&const_mem[SR_GRID_DIM_X]) = static_cast<uint32_t>(gridDim);
            *reinterpret_cast<uint32_t*>(&const_mem[SR_CTAID_X])    = static_cast<uint32_t>(cta);
            *reinterpret_cast<uint32_t*>(&const_mem[SR_TID_X])      = static_cast<uint32_t>(tid);
            *reinterpret_cast<uint32_t*>(&const_mem[SR_LANE_ID])    = static_cast<uint32_t>(tid & 31);

            func();
        }
    }
}
