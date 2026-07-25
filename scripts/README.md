# Development scripts

Reviewer reproduction is defined only by
[`reproduce/README.md`](../reproduce/README.md). The files here are
development utilities and are not called by the artifact workflow.

- `utils/compile_ll_arm.py` emits an AArch64 object from lifted LLVM IR.
- `utils/extract_kernel_sass.py` slices requested symbols from a library SASS
  dump.
- `utils/gen_addr_mutations.py` generates the development address-pattern
  fixture family.
- `utils/split_sass.py` splits every function in a multi-function SASS dump.

`hecbench/` holds the HeCBench CPU execution harness and its tracked
driver contracts (`driver_hints.json`), which the ablation runner reads.
