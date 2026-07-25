#pragma once
// SM90 x86 runtime — same as SM75 but params start at c[0][0x210]
// (SM90 kernel parameters begin after the 8-byte TMA descriptor at 0x208)
#define SLIFTER_ARG_BASE 0x210
#include "kernel_wrapper_75.h"
