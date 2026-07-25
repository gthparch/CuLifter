#!/usr/bin/env python3
"""Generate mutation variants of seed address-pattern kernels.

Pipeline: generate .cu + ref_host.cu → compile SM90 cubin → dump SASS →
extract address features → compute skeleton → deduplicate → report.

Usage:
    python scripts/utils/gen_addr_mutations.py --sm 90
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "test" / "addrtest"
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# SASS Feature Extraction
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SASSFeatures:
    imad_wide: bool
    iadd3x: bool
    lea: bool
    lea_hi: bool
    lea_hi_x: bool
    shf_64: bool
    signed_disp: bool
    loaded_base: bool  # LDG.E.64 present (pointer load)
    store_carry: bool  # IADD3.X present AND STG present
    n_ldg: int         # bucketed: 1, 2, 3, 4, 5+
    n_stg: int         # bucketed: 1, 2+

    @property
    def skeleton_key(self) -> tuple:
        """Hashable key for deduplication (booleans only)."""
        return (self.imad_wide, self.iadd3x, self.lea, self.lea_hi,
                self.lea_hi_x, self.shf_64, self.signed_disp,
                self.loaded_base, self.store_carry,
                min(self.n_ldg, 5), min(self.n_stg, 2))

    def diff(self, other: "SASSFeatures") -> int:
        """Number of boolean features that differ."""
        a, b = self.skeleton_key, other.skeleton_key
        return sum(1 for x, y in zip(a, b) if x != y)

    def feature_str(self) -> str:
        parts = []
        if self.imad_wide: parts.append("IMAD.W")
        if self.iadd3x: parts.append("IADD3.X")
        if self.lea: parts.append("LEA")
        if self.lea_hi: parts.append("LEA.HI")
        if self.lea_hi_x: parts.append("LEA.HI.X")
        if self.shf_64: parts.append("SHF64")
        if self.signed_disp: parts.append("±disp")
        if self.loaded_base: parts.append("ldbase")
        if self.store_carry: parts.append("st_carry")
        return " ".join(parts) if parts else "(none)"


def extract_features(sass_text: str) -> SASSFeatures:
    return SASSFeatures(
        imad_wide=bool(re.search(r'IMAD\.WIDE', sass_text)),
        iadd3x=bool(re.search(r'IADD3\.X', sass_text)),
        lea=bool(re.search(r'\bLEA\s', sass_text)),
        lea_hi=bool(re.search(r'LEA\.HI\b(?!\.)', sass_text)),
        lea_hi_x=bool(re.search(r'LEA\.HI\.X', sass_text)),
        shf_64=bool(re.search(r'SHF\.\w+\.(U64|S64)', sass_text) or
                     re.search(r'SHF\.\w+\.S32\.HI', sass_text)),
        signed_disp=bool(re.search(r'(?:LDG|STG)\S*.*\[.*\+\-', sass_text)),
        loaded_base=bool(re.search(r'LDG\.E\.64', sass_text)),
        store_carry=(bool(re.search(r'IADD3\.X', sass_text)) and
                     bool(re.search(r'STG', sass_text))),
        n_ldg=len(re.findall(r'\bLDG\b', sass_text)),
        n_stg=len(re.findall(r'\bSTG\b', sass_text)),
    )


def count_opcodes(sass_text: str) -> Dict[str, int]:
    """Count specific address-related opcodes."""
    ops = {}
    for pat in ["IMAD.WIDE", "IADD3.X", "IADD3(?!.X)", "LEA.HI.X",
                "LEA.HI(?!.X)", r"LEA(?!\.)", "SHF"]:
        ops[pat.replace("(?!.X)", "").replace(r"(?!\.)", "")] = len(re.findall(pat, sass_text))
    return ops


# ═══════════════════════════════════════════════════════════════════════
# Parameterized Kernel Generators
# ═══════════════════════════════════════════════════════════════════════

def _ref_host(kernel_def: str, kname: str, setup: str, launch: str,
              copy_back: str, verify: str, cleanup: str) -> str:
    """Assemble ref_host.cu from pieces."""
    return f"""\
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

{kernel_def}

int main(int argc, char** argv) {{
{setup}
{launch}
    cudaDeviceSynchronize();
{copy_back}

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {{
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }}

    bool ok = true;
{verify}
    printf(ok ? "TEST PASSED\\n" : "FAIL\\n");

{cleanup}
    return ok ? 0 : 1;
}}
"""


# --- H2 family: large-stride 64-bit index ---

def gen_H2_variant(name: str, idx_type: str = "size_t", big: str = "1048576ULL",
                   elem: str = "float", N: int = 1024) -> Tuple[str, str]:
    sz = {"float": "sizeof(float)", "double": "sizeof(double)", "int": "sizeof(int)"}[elem]
    scale = {"float": "0x4", "double": "0x8", "int": "0x4"}[elem]

    kernel_def = f"""\
extern "C" __global__ void {name}(const {elem}* in, {elem}* out, int N) {{
    {idx_type} idx = ({idx_type})blockIdx.x * {big} + ({idx_type})threadIdx.x;
    int i = (int)(idx % ({idx_type})N);
    if (i < N)
        out[i] = in[i];
}}"""

    cu = kernel_def + "\n"

    blk = min(N, 256)
    init = f"({elem})((i % 7 + 1) * 0.1f)" if elem != "int" else "(i % 7 + 1)"
    setup = f"""\
    constexpr int N = {N};
    {elem} h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = {init};
    {elem} *d_in, *d_out;
    cudaMalloc(&d_in, N * {sz}); cudaMalloc(&d_out, N * {sz});
    cudaMemcpy(d_in, h_in, N * {sz}, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * {sz});"""
    launch = f"    {name}<<<1, {blk}>>>(d_in, d_out, N);"
    copy_back = f"    cudaMemcpy(h_out, d_out, N * {sz}, cudaMemcpyDeviceToHost);"
    verify = f"""\
    for (int i = 0; i < {blk}; i++) {{
        if (h_out[i] != h_in[i]) {{
            printf("FAIL at %d\\n", i); ok = false; break;
        }}
    }}"""
    cleanup = f"    cudaFree(d_in); cudaFree(d_out);"

    ref = _ref_host(kernel_def, name, setup, launch, copy_back, verify, cleanup)
    return cu, ref


# --- H4 family: 2D pitch ---

def gen_H4_variant(name: str, idx_type: str = "size_t", elem: str = "float",
                   W: int = 32, H: int = 32) -> Tuple[str, str]:
    sz = {"float": "sizeof(float)", "double": "sizeof(double)", "int": "sizeof(int)"}[elem]

    kernel_def = f"""\
extern "C" __global__ void {name}(const {elem}* in, {elem}* out,
                                   {idx_type} pitch, int W, int H) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row < H) {{
        {idx_type} idx = ({idx_type})row * pitch + ({idx_type})col;
        out[idx] = in[idx];
    }}
}}"""

    cu = kernel_def + "\n"
    total = W * H
    init = f"({elem})((i % 7 + 1) * 0.1f)" if elem != "int" else "(i % 7 + 1)"
    setup = f"""\
    constexpr int W = {W}, H = {H}, total = W * H;
    {elem} h_in[{total}], h_out[{total}];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < total; i++) h_in[i] = {init};
    {elem} *d_in, *d_out;
    cudaMalloc(&d_in, total * {sz}); cudaMalloc(&d_out, total * {sz});
    cudaMemcpy(d_in, h_in, total * {sz}, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, total * {sz});"""
    launch = f"""\
    dim3 block(16, 16); dim3 grid((W+15)/16, (H+15)/16);
    {name}<<<grid, block>>>(d_in, d_out, ({idx_type})W, W, H);"""
    copy_back = f"    cudaMemcpy(h_out, d_out, total * {sz}, cudaMemcpyDeviceToHost);"
    verify = f"""\
    for (int i = 0; i < total; i++) {{
        if (h_out[i] != h_in[i]) {{
            printf("FAIL at %d\\n", i); ok = false; break;
        }}
    }}"""
    cleanup = f"    cudaFree(d_in); cudaFree(d_out);"

    ref = _ref_host(kernel_def, name, setup, launch, copy_back, verify, cleanup)
    return cu, ref


# --- X1 family: 2D stencil + pitch ---

def gen_X1_variant(name: str, pitch_type: str = "size_t", elem: str = "float",
                   radius: int = 1, positive_only: bool = False,
                   W: int = 32, H: int = 32) -> Tuple[str, str]:
    sz = {"float": "sizeof(float)", "double": "sizeof(double)"}[elem]
    abs_fn = "fabsf" if elem == "float" else "fabs"
    tol = "1e-5f" if elem == "float" else "1e-10"

    # Build neighbor list
    neighbors = []
    if not positive_only:
        neighbors.append(("n", f"({pitch_type})(row - {radius}) * pitch + ({pitch_type})col"))
        neighbors.append(("w", f"({pitch_type})row * pitch + ({pitch_type})(col - {radius})"))
    neighbors.append(("s", f"({pitch_type})(row + {radius}) * pitch + ({pitch_type})col"))
    neighbors.append(("e", f"({pitch_type})row * pitch + ({pitch_type})(col + {radius})"))
    neighbors.append(("c", f"({pitch_type})row * pitch + ({pitch_type})col"))

    loads = "\n".join(f"        {elem} v_{tag} = in[{expr}];" for tag, expr in neighbors)
    sumexpr = " + ".join(f"v_{tag}" for tag, _ in neighbors)

    r = radius
    guard = f"col >= {r} && col < W - {r} && row >= {r} && row < H - {r}"

    kernel_def = f"""\
extern "C" __global__ void {name}(const {elem}* in, {elem}* out,
                                   {pitch_type} pitch, int W, int H) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ({guard}) {{
{loads}
        {pitch_type} oc = ({pitch_type})row * pitch + ({pitch_type})col;
        out[oc] = {sumexpr};
    }}
}}"""

    cu = kernel_def + "\n"
    total = W * H
    init = f"({elem})((i % 7 + 1) * 0.1f)" if elem == "float" else f"({elem})((i % 7 + 1) * 0.1)"

    # Build CPU verification
    cpu_neighbors = []
    if not positive_only:
        cpu_neighbors.append(f"h_in[(row-{r})*W + col]")
        cpu_neighbors.append(f"h_in[row*W + (col-{r})]")
    cpu_neighbors.append(f"h_in[(row+{r})*W + col]")
    cpu_neighbors.append(f"h_in[row*W + (col+{r})]")
    cpu_neighbors.append(f"h_in[row*W + col]")
    cpu_sum = " + ".join(cpu_neighbors)

    setup = f"""\
    constexpr int W = {W}, H = {H}, total = W * H;
    {elem} h_in[{total}], h_out[{total}];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < total; i++) h_in[i] = {init};
    {elem} *d_in, *d_out;
    cudaMalloc(&d_in, total * {sz}); cudaMalloc(&d_out, total * {sz});
    cudaMemcpy(d_in, h_in, total * {sz}, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, total * {sz});"""
    launch = f"""\
    dim3 block(16, 16); dim3 grid((W+15)/16, (H+15)/16);
    {name}<<<grid, block>>>(d_in, d_out, ({pitch_type})W, W, H);"""
    copy_back = f"    cudaMemcpy(h_out, d_out, total * {sz}, cudaMemcpyDeviceToHost);"
    verify = f"""\
    for (int row = {r}; row < H - {r}; row++) {{
        for (int col = {r}; col < W - {r}; col++) {{
            int c = row * W + col;
            {elem} expected = {cpu_sum};
            if ({abs_fn}(h_out[c] - expected) > {tol}) {{
                printf("FAIL at row=%d col=%d\\n", row, col);
                ok = false; goto done;
            }}
        }}
    }}
    done:"""
    cleanup = f"    cudaFree(d_in); cudaFree(d_out);"

    ref = _ref_host(kernel_def, name, setup, launch, copy_back, verify, cleanup)
    return cu, ref


# --- M2 family: pointer array ---

def gen_M2_variant(name: str, elem: str = "float", idx_type: str = "int",
                   nptrs: int = 3, double_indirect: bool = False,
                   N: int = 1024) -> Tuple[str, str]:
    sz = {"float": "sizeof(float)", "double": "sizeof(double)", "int": "sizeof(int)"}[elem]
    blk = min(N, 256)

    if not double_indirect:
        kernel_def = f"""\
extern "C" __global__ void {name}(const {elem}* const* ptrs, {elem}* out,
                                   int N, int which) {{
    {idx_type} i = ({idx_type})blockIdx.x * ({idx_type})blockDim.x + ({idx_type})threadIdx.x;
    if (i < ({idx_type})N) {{
        const {elem}* src = ptrs[which];
        out[i] = src[i];
    }}
}}"""
    else:
        kernel_def = f"""\
extern "C" __global__ void {name}(const {elem}* const* const* ptr_tables,
                                   {elem}* out, int N, int table, int which) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {{
        const {elem}* const* ptrs = ptr_tables[table];
        const {elem}* src = ptrs[which];
        out[i] = src[i];
    }}
}}"""

    cu = kernel_def + "\n"
    init = f"({elem})((i % 7 + 1 + p) * 0.1f)" if elem != "int" else "(i % 7 + 1 + p)"

    if not double_indirect:
        setup = f"""\
    constexpr int N = {N}, NPTRS = {nptrs};
    {elem} *h_bufs[NPTRS], *d_bufs[NPTRS];
    for (int p = 0; p < NPTRS; p++) {{
        h_bufs[p] = new {elem}[N];
        for (int i = 0; i < N; i++) h_bufs[p][i] = {init};
        cudaMalloc(&d_bufs[p], N * {sz});
        cudaMemcpy(d_bufs[p], h_bufs[p], N * {sz}, cudaMemcpyHostToDevice);
    }}
    const {elem}* h_ptrs[NPTRS];
    for (int p = 0; p < NPTRS; p++) h_ptrs[p] = d_bufs[p];
    const {elem}** d_ptrs;
    cudaMalloc(&d_ptrs, NPTRS * sizeof(const {elem}*));
    cudaMemcpy(d_ptrs, h_ptrs, NPTRS * sizeof(const {elem}*), cudaMemcpyHostToDevice);
    {elem} h_out[N], *d_out;
    cudaMalloc(&d_out, N * {sz}); cudaMemset(d_out, 0, N * {sz});"""
        launch = f"    int which = 1;\n    {name}<<<(N+{blk-1})/{blk}, {blk}>>>(d_ptrs, d_out, N, which);"
        verify = f"""\
    for (int i = 0; i < N; i++) {{
        if (h_out[i] != h_bufs[which][i]) {{
            printf("FAIL at %d\\n", i); ok = false; break;
        }}
    }}"""
        cleanup = f"""\
    for (int p = 0; p < NPTRS; p++) {{ cudaFree(d_bufs[p]); delete[] h_bufs[p]; }}
    cudaFree(d_ptrs); cudaFree(d_out);"""
    else:
        setup = f"""\
    constexpr int N = {N}, NTABLES = 2, NPTRS = {nptrs};
    {elem} *h_bufs[NTABLES][NPTRS], *d_bufs[NTABLES][NPTRS];
    for (int t = 0; t < NTABLES; t++)
        for (int p = 0; p < NPTRS; p++) {{
            h_bufs[t][p] = new {elem}[N];
            for (int i = 0; i < N; i++) h_bufs[t][p][i] = ({elem})((i % 7 + 1 + t + p) * 0.1f);
            cudaMalloc(&d_bufs[t][p], N * {sz});
            cudaMemcpy(d_bufs[t][p], h_bufs[t][p], N * {sz}, cudaMemcpyHostToDevice);
        }}
    const {elem}* h_ptrs[NTABLES][NPTRS];
    for (int t = 0; t < NTABLES; t++)
        for (int p = 0; p < NPTRS; p++) h_ptrs[t][p] = d_bufs[t][p];
    const {elem}** d_ptrs[NTABLES];
    for (int t = 0; t < NTABLES; t++) {{
        cudaMalloc(&d_ptrs[t], NPTRS * sizeof(const {elem}*));
        cudaMemcpy(d_ptrs[t], h_ptrs[t], NPTRS * sizeof(const {elem}*), cudaMemcpyHostToDevice);
    }}
    const {elem}** h_tables[NTABLES];
    for (int t = 0; t < NTABLES; t++) h_tables[t] = d_ptrs[t];
    const {elem}*** d_tables;
    cudaMalloc(&d_tables, NTABLES * sizeof(const {elem}**));
    cudaMemcpy(d_tables, h_tables, NTABLES * sizeof(const {elem}**), cudaMemcpyHostToDevice);
    {elem} h_out[N], *d_out;
    cudaMalloc(&d_out, N * {sz}); cudaMemset(d_out, 0, N * {sz});"""
        launch = f"""\
    int table = 0, which = 1;
    {name}<<<(N+{blk-1})/{blk}, {blk}>>>(d_tables, d_out, N, table, which);"""
        verify = f"""\
    for (int i = 0; i < N; i++) {{
        if (h_out[i] != h_bufs[table][which][i]) {{
            printf("FAIL at %d\\n", i); ok = false; break;
        }}
    }}"""
        cleanup = f"""\
    for (int t = 0; t < NTABLES; t++) {{
        for (int p = 0; p < NPTRS; p++) {{ cudaFree(d_bufs[t][p]); delete[] h_bufs[t][p]; }}
        cudaFree(d_ptrs[t]);
    }}
    cudaFree(d_tables); cudaFree(d_out);"""

    copy_back = f"    cudaMemcpy(h_out, d_out, N * {sz}, cudaMemcpyDeviceToHost);"
    ref = _ref_host(kernel_def, name, setup, launch, copy_back, verify, cleanup)
    return cu, ref


# --- X2 family: pointer array + stencil ---

def gen_X2_variant(name: str, elem: str = "float", idx_type: str = "int",
                   radius: int = 1, N: int = 1024) -> Tuple[str, str]:
    sz = {"float": "sizeof(float)", "double": "sizeof(double)"}[elem]
    abs_fn = "fabsf" if elem == "float" else "fabs"
    tol = "1e-5f" if elem == "float" else "1e-10"
    blk = min(N, 256)

    # Build stencil accesses
    offsets = list(range(-radius, radius + 1))
    loads = "\n".join(
        f"        {elem} v{i} = src[i + ({idx_type})({o})];"
        if o != 0 else f"        {elem} v{i} = src[i];"
        for i, o in enumerate(offsets)
    )
    sumexpr = " + ".join(f"v{i}" for i in range(len(offsets)))

    kernel_def = f"""\
extern "C" __global__ void {name}(const {elem}* const* ptrs, {elem}* out,
                                   int N, int which) {{
    {idx_type} i = ({idx_type})blockIdx.x * ({idx_type})blockDim.x + ({idx_type})threadIdx.x;
    if (i >= {radius} && i < ({idx_type})(N - {radius})) {{
        const {elem}* src = ptrs[which];
{loads}
        out[i] = {sumexpr};
    }}
}}"""

    cu = kernel_def + "\n"
    init = f"({elem})((i % 7 + 1 + p) * 0.1f)" if elem != "int" else "(i % 7 + 1 + p)"

    cpu_sum = " + ".join(f"h_bufs[which][i+({o})]" for o in offsets)

    setup = f"""\
    constexpr int N = {N}, NPTRS = 2;
    {elem} *h_bufs[NPTRS], *d_bufs[NPTRS];
    for (int p = 0; p < NPTRS; p++) {{
        h_bufs[p] = new {elem}[N];
        for (int i = 0; i < N; i++) h_bufs[p][i] = {init};
        cudaMalloc(&d_bufs[p], N * {sz});
        cudaMemcpy(d_bufs[p], h_bufs[p], N * {sz}, cudaMemcpyHostToDevice);
    }}
    const {elem}* h_ptrs[NPTRS];
    for (int p = 0; p < NPTRS; p++) h_ptrs[p] = d_bufs[p];
    const {elem}** d_ptrs;
    cudaMalloc(&d_ptrs, NPTRS * sizeof(const {elem}*));
    cudaMemcpy(d_ptrs, h_ptrs, NPTRS * sizeof(const {elem}*), cudaMemcpyHostToDevice);
    {elem} h_out[N], *d_out;
    cudaMalloc(&d_out, N * {sz}); cudaMemset(d_out, 0, N * {sz});"""
    launch = f"    int which = 0;\n    {name}<<<(N+{blk-1})/{blk}, {blk}>>>(d_ptrs, d_out, N, which);"
    copy_back = f"    cudaMemcpy(h_out, d_out, N * {sz}, cudaMemcpyDeviceToHost);"
    verify = f"""\
    for (int i = {radius}; i < N - {radius}; i++) {{
        {elem} expected = {cpu_sum};
        if ({abs_fn}(h_out[i] - expected) > {tol}) {{
            printf("FAIL at %d\\n", i); ok = false; break;
        }}
    }}"""
    cleanup = f"""\
    for (int p = 0; p < NPTRS; p++) {{ cudaFree(d_bufs[p]); delete[] h_bufs[p]; }}
    cudaFree(d_ptrs); cudaFree(d_out);"""

    ref = _ref_host(kernel_def, name, setup, launch, copy_back, verify, cleanup)
    return cu, ref


# ═══════════════════════════════════════════════════════════════════════
# Mutation Definitions
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Mutation:
    name: str
    seed: str
    knob: str       # which knob was varied
    tags: List[str]
    gen_fn: str     # generator function name
    gen_kwargs: dict
    opt_level: str = "-O2"
    description: str = ""


MUTATIONS: List[Mutation] = [
    # --- H2 family: vary index type, multiplier, element type ---
    Mutation("H2m_idx_int", "H2_big_mul_sizet", "index_type",
             ["64bit_index", "carry_sensitive"],
             "gen_H2_variant", {"idx_type": "int", "big": "1048576ULL"},
             description="Control: int index drops carry chain"),
    Mutation("H2m_idx_ll", "H2_big_mul_sizet", "index_type",
             ["64bit_index", "carry_sensitive"],
             "gen_H2_variant", {"idx_type": "long long", "big": "1048576LL"},
             description="Signed 64-bit index"),
    Mutation("H2m_idx_ull", "H2_big_mul_sizet", "index_type",
             ["64bit_index", "carry_sensitive"],
             "gen_H2_variant", {"idx_type": "unsigned long long", "big": "1048576ULL"},
             description="Explicit unsigned 64-bit"),
    Mutation("H2m_big28", "H2_big_mul_sizet", "multiplier",
             ["64bit_index", "carry_sensitive"],
             "gen_H2_variant", {"idx_type": "size_t", "big": "268435456ULL"},
             description="1<<28 multiplier, more carry pressure"),
    Mutation("H2m_double", "H2_big_mul_sizet", "elem_type",
             ["64bit_index", "carry_sensitive"],
             "gen_H2_variant", {"idx_type": "size_t", "elem": "double"},
             description="Double element, wider stride"),
    Mutation("H2m_O0", "H2_big_mul_sizet", "opt_level",
             ["64bit_index", "carry_sensitive"],
             "gen_H2_variant", {"idx_type": "size_t"},
             opt_level="-O0", description="No optimization"),

    # --- H4 family: vary index type, pitch, element type ---
    Mutation("H4m_idx_int", "H4_pitch_2d_sizet", "index_type",
             ["64bit_index"],
             "gen_H4_variant", {"idx_type": "int"},
             description="Control: int index"),
    Mutation("H4m_idx_ull", "H4_pitch_2d_sizet", "index_type",
             ["64bit_index", "carry_sensitive"],
             "gen_H4_variant", {"idx_type": "unsigned long long"},
             description="Explicit unsigned long long"),
    Mutation("H4m_double", "H4_pitch_2d_sizet", "elem_type",
             ["64bit_index"],
             "gen_H4_variant", {"idx_type": "size_t", "elem": "double"},
             description="Double element"),
    Mutation("H4m_large", "H4_pitch_2d_sizet", "pitch_magnitude",
             ["64bit_index", "carry_sensitive"],
             "gen_H4_variant", {"idx_type": "size_t", "W": 64, "H": 64},
             description="Larger dimensions (64x64)"),

    # --- X1 family: vary pitch type, radius, element, offset style ---
    Mutation("X1m_pitch_int", "X1_stencil_64pitch_float", "pitch_type",
             ["signed_disp"],
             "gen_X1_variant", {"pitch_type": "int"},
             description="Control: int pitch drops 64-bit math"),
    Mutation("X1m_radius2", "X1_stencil_64pitch_float", "stencil_radius",
             ["signed_disp", "64bit_index"],
             "gen_X1_variant", {"pitch_type": "size_t", "radius": 2},
             description="Radius-2 stencil, wider displacements"),
    Mutation("X1m_double", "X1_stencil_64pitch_float", "elem_type",
             ["signed_disp", "64bit_index"],
             "gen_X1_variant", {"pitch_type": "size_t", "elem": "double"},
             description="Double element, 8-byte displacements"),
    Mutation("X1m_pos_only", "X1_stencil_64pitch_float", "offset_style",
             ["64bit_index"],
             "gen_X1_variant", {"pitch_type": "size_t", "positive_only": True},
             description="Positive-only offsets (no north/west)"),
    Mutation("X1m_large", "X1_stencil_64pitch_float", "dimensions",
             ["signed_disp", "64bit_index"],
             "gen_X1_variant", {"pitch_type": "size_t", "W": 64, "H": 64},
             description="Larger grid 64x64"),

    # --- M2 family: vary element, index, indirection depth ---
    Mutation("M2m_double", "M2_ptr_array_float", "elem_type",
             ["ptr_indirect"],
             "gen_M2_variant", {"elem": "double"},
             description="Double element, 64-bit payload loads"),
    Mutation("M2m_idx_sizet", "M2_ptr_array_float", "index_type",
             ["ptr_indirect", "64bit_index"],
             "gen_M2_variant", {"idx_type": "size_t"},
             description="size_t payload index adds carry chain"),
    Mutation("M2m_dbl_indirect", "M2_ptr_array_float", "indirection_depth",
             ["ptr_indirect"],
             "gen_M2_variant", {"double_indirect": True},
             description="Double indirection: table → ptrs → data"),
    Mutation("M2m_8ptrs", "M2_ptr_array_float", "num_ptrs",
             ["ptr_indirect"],
             "gen_M2_variant", {"nptrs": 8},
             description="8 pointers in array"),

    # --- X2 family: vary radius, element, index type ---
    Mutation("X2m_radius2", "X2_ptr_stencil_float", "stencil_radius",
             ["ptr_indirect", "signed_disp"],
             "gen_X2_variant", {"radius": 2},
             description="Radius-2 stencil after indirection"),
    Mutation("X2m_double", "X2_ptr_stencil_float", "elem_type",
             ["ptr_indirect", "signed_disp"],
             "gen_X2_variant", {"elem": "double"},
             description="Double element, wider displacements"),
    Mutation("X2m_idx_sizet", "X2_ptr_stencil_float", "index_type",
             ["ptr_indirect", "signed_disp", "64bit_index"],
             "gen_X2_variant", {"idx_type": "size_t"},
             description="size_t index + stencil + indirection"),
    Mutation("X2m_radius4", "X2_ptr_stencil_float", "stencil_radius",
             ["ptr_indirect", "signed_disp"],
             "gen_X2_variant", {"radius": 4},
             description="Radius-4 stencil, many displacements"),
]


# ═══════════════════════════════════════════════════════════════════════
# Build Pipeline
# ═══════════════════════════════════════════════════════════════════════

GENERATORS = {
    "gen_H2_variant": gen_H2_variant,
    "gen_H4_variant": gen_H4_variant,
    "gen_X1_variant": gen_X1_variant,
    "gen_M2_variant": gen_M2_variant,
    "gen_X2_variant": gen_X2_variant,
}


def run_cmd(cmd, timeout=60, cwd=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def find_tool(name: str) -> str:
    for p in [f"/usr/local/cuda/bin/{name}"]:
        if Path(p).exists():
            return p
    return name


def generate_and_build(mut: Mutation, output_dir: Path, sm: int) -> Optional[SASSFeatures]:
    """Generate, compile, extract SASS, return features (or None on failure)."""
    gen_fn = GENERATORS[mut.gen_fn]
    kwargs = dict(mut.gen_kwargs)
    kwargs["name"] = mut.name

    cu_code, ref_host_code = gen_fn(**kwargs)

    kdir = output_dir / mut.name
    kdir.mkdir(parents=True, exist_ok=True)

    cu_path = kdir / f"{mut.name}.cu"
    cu_path.write_text(cu_code)
    (kdir / "ref_host.cu").write_text(ref_host_code)

    # Save metadata
    meta = {
        "name": mut.name, "seed": mut.seed, "knob": mut.knob,
        "tags": mut.tags, "opt": mut.opt_level, "description": mut.description,
        "gen_kwargs": {k: str(v) for k, v in mut.gen_kwargs.items()},
    }

    nvcc = find_tool("nvcc")
    cuobjdump = find_tool("cuobjdump")
    nvdisasm_tool = find_tool("nvdisasm")
    clean_env = dict(os.environ)
    clean_env["PATH"] = f"/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    # 1. Compile cubin
    cubin = kdir / f"{mut.name}.cubin"
    rc, _, err = run_cmd([nvcc, "--cubin", str(cu_path), "-o", str(cubin),
                          f"-arch=sm_{sm}", mut.opt_level])
    if rc != 0:
        meta["build_status"] = f"cubin_fail: {err.strip().split(chr(10))[-1]}"
        (kdir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        return None

    # 2. PTX
    ptx = kdir / f"{mut.name}.ptx"
    run_cmd([nvcc, "-ptx", str(cu_path), "-o", str(ptx), f"-arch=sm_{sm}", mut.opt_level])

    # 3. SASS
    sass_path = kdir / f"{mut.name}_sm{sm}.sass"
    rc, out, _ = run_cmd([cuobjdump, "--dump-sass", str(cubin)])
    if rc == 0:
        sass_path.write_text(out)

    # 4. ELF
    elf_path = kdir / f"{mut.name}_sm{sm}.elf"
    rc, out, _ = run_cmd([cuobjdump, "--dump-elf", str(cubin)])
    if rc == 0:
        elf_path.write_text(out)

    # 5. nvdisasm
    nvdis = kdir / f"{mut.name}.nvdisasm"
    rc, out, _ = run_cmd([nvdisasm_tool, str(cubin)])
    if rc == 0:
        nvdis.write_text(out)

    # 6. Extract features
    sass_text = sass_path.read_text() if sass_path.exists() else ""
    features = extract_features(sass_text)
    opcounts = count_opcodes(sass_text)

    meta["build_status"] = "ok"
    meta["sass_lines"] = len(sass_text.splitlines())
    meta["features"] = asdict(features)
    meta["opcode_counts"] = opcounts
    meta["skeleton_key"] = list(features.skeleton_key)
    (kdir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    return features


def verify_on_gpu(mut: Mutation, output_dir: Path, sm: int) -> Tuple[bool, str]:
    """Compile and run ref_host.cu on GPU."""
    kdir = output_dir / mut.name
    ref = kdir / "ref_host.cu"
    if not ref.exists():
        return False, "no ref_host.cu"

    nvcc = find_tool("nvcc")
    exe = kdir / "ref_test"
    # Use sm_80 for A100 execution (sm_90 won't run on A100)
    rc, _, err = run_cmd([nvcc, str(ref), "-o", str(exe), "-arch=sm_80"],
                         timeout=30)
    if rc != 0:
        return False, f"compile: {err.strip().split(chr(10))[-1]}"

    rc, out, err = run_cmd([str(exe)], cwd=str(kdir), timeout=30)
    if "TEST PASSED" in out:
        # Also dump ref output
        run_cmd([str(exe), "--dump"], cwd=str(kdir), timeout=30)
        return True, "PASS"
    return False, f"rc={rc}: {out.strip().split(chr(10))[-1] if out.strip() else err.strip()}"


# ═══════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════

def deduplicate(results: Dict[str, SASSFeatures],
                seed_features: Dict[str, SASSFeatures]) -> Dict[str, str]:
    """Filter mutations: keep those with new skeletons or minimal neighbor pairs.

    Returns dict of name → reason (why kept).
    """
    all_features = {}
    all_features.update(seed_features)
    all_features.update(results)

    # Existing skeletons from seeds
    seen_skeletons = {name: feat.skeleton_key for name, feat in seed_features.items()}

    kept = {}  # name → reason

    for name, feat in sorted(results.items()):
        sk = feat.skeleton_key

        # Check if this is a new skeleton
        is_new = sk not in set(seen_skeletons.values())

        # Check if this forms a minimal neighbor pair with any seed
        neighbor_of = None
        for seed_name, seed_feat in seed_features.items():
            if feat.diff(seed_feat) == 1:
                neighbor_of = seed_name
                break

        # Also check neighbor with other kept mutations
        if neighbor_of is None:
            for kept_name in kept:
                if kept_name in all_features and feat.diff(all_features[kept_name]) == 1:
                    neighbor_of = kept_name
                    break

        if is_new:
            kept[name] = f"new skeleton {sk}"
            seen_skeletons[name] = sk
        elif neighbor_of:
            kept[name] = f"neighbor of {neighbor_of} (diff=1)"
            seen_skeletons[name] = sk
        # else: duplicate skeleton AND no neighbor pair → skip

    return kept


# ═══════════════════════════════════════════════════════════════════════
# Load seed features from existing kernels
# ═══════════════════════════════════════════════════════════════════════

def load_seed_features(output_dir: Path, sm: int) -> Dict[str, SASSFeatures]:
    """Load features from existing seed kernels."""
    seeds = {}
    seed_names = ["H2_big_mul_sizet", "H4_pitch_2d_sizet", "X1_stencil_64pitch_float",
                  "M2_ptr_array_float", "X2_ptr_stencil_float",
                  # Also include the control variants
                  "H2_big_mul_int", "H4_pitch_2d_int",
                  "A1_copy_1d_float", "N1_stencil_3pt_float", "N2_stencil_5pt_float",
                  "M1_select_two_float", "X4_base_large_off_float"]

    for name in seed_names:
        sass_path = output_dir / name / f"{name}_sm{sm}.sass"
        if sass_path.exists():
            seeds[name] = extract_features(sass_path.read_text())
    return seeds


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate mutation variants and analyze SASS")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--sm", type=int, default=90)
    parser.add_argument("--verify", action="store_true", help="Run ref_host.cu on GPU")
    parser.add_argument("--keep-all", action="store_true", help="Skip deduplication")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Load seed features
    print("Loading seed kernel features...")
    seed_features = load_seed_features(output_dir, args.sm)
    print(f"  Found {len(seed_features)} seed kernels")
    for name, feat in sorted(seed_features.items()):
        print(f"    {name:35s} {feat.feature_str()}")

    # Generate and build all mutations
    print(f"\nGenerating {len(MUTATIONS)} mutations...")
    results = {}
    for mut in MUTATIONS:
        feat = generate_and_build(mut, output_dir, args.sm)
        if feat:
            results[mut.name] = feat
            print(f"  OK  {mut.name:30s} {feat.feature_str()}")
        else:
            print(f"  FAIL {mut.name:30s}")

    # GPU verification
    if args.verify:
        print(f"\nVerifying {len(MUTATIONS)} mutations on GPU...")
        pass_count = 0
        for mut in MUTATIONS:
            ok, detail = verify_on_gpu(mut, output_dir, args.sm)
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {mut.name}: {detail}")
            if ok:
                pass_count += 1
                # Update metadata
                meta_path = output_dir / mut.name / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    meta["gpu_ref"] = {"status": "pass" if ok else "fail", "detail": detail}
                    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"\n  GPU verification: {pass_count}/{len(MUTATIONS)}")

    # Deduplication
    if not args.keep_all:
        print(f"\nDeduplicating by SASS skeleton...")
        kept = deduplicate(results, seed_features)
        dropped = set(results.keys()) - set(kept.keys())

        print(f"\n  Kept: {len(kept)}/{len(results)} mutations")
        for name, reason in sorted(kept.items()):
            feat = results[name]
            print(f"    KEEP  {name:30s} {feat.feature_str():40s} | {reason}")

        if dropped:
            print(f"\n  Dropped {len(dropped)} duplicates:")
            for name in sorted(dropped):
                feat = results[name]
                print(f"    DROP  {name:30s} {feat.feature_str()}")

        # Mark dropped variants in metadata
        for name in dropped:
            meta_path = output_dir / name / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["dedup_status"] = "dropped"
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        for name, reason in kept.items():
            meta_path = output_dir / name / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["dedup_status"] = f"kept: {reason}"
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    else:
        kept = {name: "keep-all" for name in results}

    # Summary table
    print(f"\n{'='*100}")
    print("FINAL INVENTORY: Seeds + Kept Mutations")
    print(f"{'='*100}")
    print(f"{'Name':35s} {'IMAD.W':>6s} {'I3.X':>5s} {'LEA':>4s} {'L.HI':>5s} {'LHX':>4s} "
          f"{'SHF':>4s} {'±dsp':>5s} {'ldbs':>5s} {'stcr':>5s} {'LDG':>4s} {'STG':>4s}")
    print("-" * 100)

    all_final = {}
    all_final.update(seed_features)
    all_final.update({k: results[k] for k in kept})

    for name in sorted(all_final):
        f = all_final[name]
        marker = "SEED" if name in seed_features else "MUT "
        print(f"[{marker}] {name:30s} "
              f"{'Y' if f.imad_wide else '.':>6s} "
              f"{'Y' if f.iadd3x else '.':>5s} "
              f"{'Y' if f.lea else '.':>4s} "
              f"{'Y' if f.lea_hi else '.':>5s} "
              f"{'Y' if f.lea_hi_x else '.':>4s} "
              f"{'Y' if f.shf_64 else '.':>4s} "
              f"{'Y' if f.signed_disp else '.':>5s} "
              f"{'Y' if f.loaded_base else '.':>5s} "
              f"{'Y' if f.store_carry else '.':>5s} "
              f"{f.n_ldg:>4d} "
              f"{f.n_stg:>4d}")

    print(f"\n  Total: {len(seed_features)} seeds + {len(kept)} mutations = {len(all_final)} kernels")

    # Neighbor pairs
    print(f"\n{'='*80}")
    print("NEIGHBOR PAIRS (differ by exactly 1 feature)")
    print(f"{'='*80}")
    names = sorted(all_final.keys())
    pairs_found = 0
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if all_final[a].diff(all_final[b]) == 1:
                pairs_found += 1
                fa, fb = all_final[a], all_final[b]
                # Find which feature differs
                ska, skb = fa.skeleton_key, fb.skeleton_key
                feat_names = ["imad_wide", "iadd3x", "lea", "lea_hi", "lea_hi_x",
                              "shf_64", "signed_disp", "loaded_base", "store_carry",
                              "n_ldg", "n_stg"]
                diff_feat = [feat_names[j] for j in range(len(ska)) if ska[j] != skb[j]]
                print(f"  {a:30s} ↔ {b:30s}  diff: {', '.join(diff_feat)}")
    if pairs_found == 0:
        print("  (none found)")
    print(f"\n  {pairs_found} neighbor pairs found")


if __name__ == "__main__":
    main()
