# cuda_sdk — CUDA SDK example kernels

**Verify method: lift-only.** No output reference exists for these example
kernels; "verification" is that the lifter emits valid IR for each one.

## Input
`elf/*.elf` — 19 sm_90 `cuobjdump --dump-elf` dumps (ELF header + symbols, one per
SDK example). Note these `--dump-elf` dumps carry **no SASS body**, so they do not
lift standalone — the real lift inputs are the SASS+ELF fixtures at
`$SLIFTER_ROOT/test/cuobjdump/<example>/`.

## Run
```bash
conda activate slifter
bash test/cuda_sdk/run.sh      # lift each kernel
bash test/cuda_sdk/verify.sh   # same — lift-only has no exec check
```
`verify.sh` execs `run.sh`; PASS = zero `LIFT-FAIL` lines.
