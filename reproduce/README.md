# CuLifter MICRO'26 artifact workflow

The reviewer workflow consumes the separately distributed `benchmark/`
archive (about 4.7 GB extracted). Start with at least 50 GB of free disk
space for the archive, extraction, and results. Every input ships with
pre-disassembled SASS and its cubin/ELF metadata.

## 1. Install

```bash
conda env create -f environment.yml -p .conda_env
export PATH="$(pwd)/.conda_env/bin:$PATH"
bash reproduce/setup_data.sh
```

`setup_data.sh` downloads the benchmark archive, verifies its checksum, and
extracts it to `benchmark/`.

## 2. Run the two artifact experiments

Run E1 followed by E2 with:

```bash
bash reproduce/run_all.sh
```

The experiments can also be run and timed separately:

```bash
# E1: lifting coverage
bash reproduce/run_cpu_sm90.sh

# E2: paper figure data
bash reproduce/figures/reproduce_figures.sh
```

`run_all.sh` accepts `--only e1` or `--only e2`. E1 also accepts:

```bash
bash reproduce/run_cpu_sm90.sh --suite cuda_sdk
bash reproduce/run_cpu_sm90.sh --suite cublas --jobs 4
```

### E1 expected result

Final console output of a full run:

```
==================== SUMMARY (CPU backend, sm90) ====================
suite                checked units   LLVM definitions
hecbench                   229/229            770/770
cublas                     177/177          4137/4137
cutlass                    105/105            641/641
flash_attention              63/63              63/63
cudnn_cnn                      7/7              46/46
cudnn_ops                    20/20          2283/2283
cudnn_adv                    30/30          4018/4018
cuda_sdk                     19/19              19/19

units lifted and verified: 650; LLVM definitions emitted: 11977
per-artifact LLVM IR, logs, and summary.csv: reproduce/results/lift_<timestamp>
total failures: 0
```

Per-artifact LLVM IR, logs, and `summary.csv` are written under
`reproduce/results/`.

### E2 expected result

E2 prints per-stage progress and finishes with a stage-time table.

```
=== done <date> ===
stage             wall_s  max_rss_kib  data_kib_after  status
collect:...       ...     ...          ...             ok
ablation          ...     ...          ...             ok
nvlib             ...     ...          ...             ok
degrade           ...     ...          ...             ok
export            ...     ...          ...             ok
```

The results are the per-figure CSVs written to
`reproduce/figures/output/`, one per paper figure (named `fig5` through
`fig11`, the filenames the paper source uses). The values we obtained are
in `reproduce/figures/expected/`.

