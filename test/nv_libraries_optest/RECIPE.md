# NVIDIA-library CPU development check

This is a developer-only execution check, separate from reviewer reproduction.
It needs a CUDA toolkit for `cuobjdump`, LLVM/Clang, a C++20 compiler, and an
untracked library corpus at:

```text
benchmark/nv_libraries/cubins/<library>.cubin
```

Activate `environment.yml`, put the external tools on `PATH`, then run:

```bash
bash test/nv_libraries_optest/verify.sh \
  TEST_NAME CUBIN_BASENAME MANGLED_NAME
```

`TEST_NAME` names a committed child directory containing `host_cpu.cpp`.
`CUBIN_BASENAME` omits `.cubin`. Set `SLIFTER_BENCHMARK_DIR` for a benchmark
root outside the repository, `CUOBJDUMP` for a nonstandard disassembler path,
or `NVLIB_LIFT_CACHE=1` to reuse an existing `kernel.ll`.

The driver must marshal SM90 kernel parameters at constant-bank base `0x210`.
By-value structs occupy consecutive parameter offsets; pointer-to-struct
parameters require real backing storage and valid nested pointers. References
must be computed independently from the lifted IR. Exact integer/data-movement
checks should use zero tolerance; floating-point checks should document their
absolute and relative tolerances.
