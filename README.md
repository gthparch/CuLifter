# CuLifter

CuLifter parses NVIDIA SASS and lifts it to LLVM IR. The implementation is
Python-based; `main.py` is the command-line entry point and supports CPU,
serialized-warp CPU, NVVM, and ARM lifter backends.

## Setup

```bash
conda env create -f environment.yml -p .conda_env
export PATH="$(pwd)/.conda_env/bin:$PATH"
python -c "import elftools, libcst, llvmlite, yaml"
```

Example:

```bash
python main.py -i input.sass input.elf -o output --lifter x86
```

`main.py` writes `output.ll`. Use `python main.py --help` for the current
options.
