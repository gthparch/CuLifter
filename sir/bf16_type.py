"""BFloat16Type — subclass of llvmlite _BaseFloatType that emits `bfloat`.

llvmlite 0.44 ships `HalfType` (f16) but not bfloat. Subclassing
_BaseFloatType and overriding __str__ is enough for IR-text emission;
LLVM 14+ accepts `bfloat` / `<N x bfloat>` in fadd/fmul/bitcast on
x86_64, aarch64 targets.

We use this type for BF16 semantic operations:
- HFMA2.BF16_V2 / HADD2.BF16_V2 / HMUL2.BF16_V2 (packed 2x bfloat)
- HMMA.BF16 / HGMMA.BF16 (tensor core BF16 matmul)

The register storage stays 32-bit (two packed bfloats); we bitcast
to `<2 x bfloat>` when doing arithmetic and back to i32 when storing.
"""

from llvmlite.ir.types import _BaseFloatType


class BFloat16Type(_BaseFloatType):
    """LLVM IR `bfloat` primitive. Intrinsic name `bf16`."""
    null = "0.0"
    intrinsic_name = "bf16"

    def __str__(self):
        return "bfloat"

    def format_constant(self, value):
        return f"{float(value):.6e}"


BFloat16Type._create_instance()
