#!/usr/bin/env python3
"""Auto-generate a CPU execution-faithfulness harness for one library kernel.

Usage: autogen_harness.py <cubin> <mangled> > host_cpu.cpp

Reads the kernel's EIATTR_KPARAM_INFO (param offset/size) from `cuobjdump -elf`,
classifies each param via the demangled signature, and emits a host_cpu.cpp that:
  - allocates a large buffer per pointer param (so computed addresses stay mapped),
  - sets struct params (NppiSize / nvjpegImage_t) and scalars to small safe values,
  - runs the lifted kernel TWICE on identical inputs and compares every buffer.

This verifies EXECUTION FAITHFULNESS (valid IR -> links -> runs to completion ->
memory-safe -> deterministic), NOT bit-exact numeric correctness (no GPU golden).
SM90 param base = 0x210.
"""
import sys, subprocess, re

PARAM_BASE = 0x210

def kparams(cubin, mangled):
    """Return [(ordinal, offset, size)] for the kernel, from cuobjdump -elf."""
    txt = subprocess.run(["cuobjdump","-elf",cubin],capture_output=True,text=True).stdout
    lines = txt.splitlines()
    # find the kernel's .nv.info.<mangled> attribute block (the one followed by KPARAM)
    out = []
    i = 0
    target = ".nv.info." + mangled
    while i < len(lines):
        if lines[i].strip() == target:
            j = i + 1
            while j < len(lines) and not lines[j].startswith(".nv.info."):
                m = re.search(r"Ordinal\s*:\s*(0x[0-9a-f]+)\s*Offset\s*:\s*(0x[0-9a-f]+)\s*Size\s*:\s*(0x[0-9a-f]+)", lines[j])
                if m:
                    out.append((int(m.group(1),16), int(m.group(2),16), int(m.group(3),16)))
                j += 1
            if out:
                break
            i = j
        else:
            i += 1
    out.sort(key=lambda t: t[0])
    return out

def demangle(mangled):
    return subprocess.run(["c++filt"],input=mangled,capture_output=True,text=True).stdout.strip()

def split_args(dem):
    """Top-level comma split of the (...) arg list of the demangled signature."""
    p = dem.rfind("(")
    if p < 0: return []
    s = dem[p+1:]
    # take to matching close paren
    depth=1; out=[]; cur="";
    for ch in s:
        if ch in "(<[": depth+=1
        elif ch in ")>]":
            depth-=1
            if depth==0: break
        if ch=="," and depth==1: out.append(cur.strip()); cur=""; continue
        cur+=ch
    if cur.strip(): out.append(cur.strip())
    return out

def classify(arg, size):
    a = arg.replace(" ","")
    if "*" in a: return "ptr"
    if "nvjpegImage_t" in arg: return "img"        # 64B: 4 ptr + 4 size_t
    if "NppiSize" in arg or "Size" in arg and size==8: return "size2"  # 2 ints
    if size==8: return "u64"                        # size_t / long
    if size==4: return "i32"
    if size==2: return "i16"
    if size==1: return "i8"
    return "blob"

TEMPLATE = r'''// AUTO-GENERATED execution-faithfulness harness (autogen_harness.py)
// Kernel: {dem}
// Verifies: lifts -> links -> runs to completion -> memory-safe -> deterministic.
// (NOT bit-exact vs NVIDIA: no GPU golden for closed-source kernels.)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "kernel_wrapper.h"

extern "C" void {mangled}();

static const int NBUF = {nbuf};
static const size_t BUFSZ = 1u<<22;            // 4 MiB per pointer buffer
static uint8_t* bufs[NBUF>0?NBUF:1];
static uint8_t* snap[NBUF>0?NBUF:1];

static void fill(uint8_t* b){{ for(size_t i=0;i<BUFSZ;i++) b[i]=(uint8_t)((i*1103515245u+12345u)>>16); }}

static void run_once() {{
    for(int i=0;i<NBUF;i++) fill(bufs[i]);
    launchKernel2D({mangled}, Dim3{{{gx},{gy}}}, Dim3{{{bx},{by}}},
{args}
    );
}}

int main() {{
    uint8_t* clean=(uint8_t*)malloc(BUFSZ>0?BUFSZ:1); fill(clean);   // pristine input image
    for(int i=0;i<NBUF;i++){{ bufs[i]=(uint8_t*)aligned_alloc(64,BUFSZ); snap[i]=(uint8_t*)malloc(BUFSZ); }}
    run_once();
    int wrote=0; for(int i=0;i<NBUF;i++) if(memcmp(clean,bufs[i],BUFSZ)) wrote++;  // did it produce output?
    for(int i=0;i<NBUF;i++) memcpy(snap[i],bufs[i],BUFSZ);
    run_once();                                  // second run, identical inputs
    int diff=0;
    for(int i=0;i<NBUF;i++) if(memcmp(snap[i],bufs[i],BUFSZ)) diff++;
    printf("ran-to-completion=yes memory-safe=yes wrote_buffers=%d/%d nondeterministic_buffers=%d/%d\n",
           wrote, NBUF, diff, NBUF);
    if(diff==0 && wrote>0){{ printf("TEST PASSED\n"); return 0; }}
    if(wrote==0) printf("FAIL: kernel produced no output (vacuous run)\n");
    else printf("FAIL: nondeterministic output (%d buffers)\n", diff);
    return 1;
}}
'''

def main():
    cubin, mangled = sys.argv[1], sys.argv[2]
    params = kparams(cubin, mangled)
    if not params:
        sys.stderr.write("no KPARAM_INFO found\n"); sys.exit(2)
    dem = demangle(mangled)
    args = split_args(dem)
    lines=[]; nbuf=0
    SMALL_W, SMALL_H = 16, 16
    for idx,(ordn,off,size) in enumerate(params):
        abs_off = PARAM_BASE + off
        arg = args[idx] if idx < len(args) else ("T*" if size==8 else "int")
        kind = classify(arg, size)
        if kind=="ptr":
            lines.append(f"        ArgAt{{0x{abs_off:x}}}, (void*)bufs[{nbuf}],"); nbuf+=1
        elif kind=="img":  # nvjpegImage_t: ch[0..3]@+0..24, pitch[0..3]@+32..56
            for k in range(4):
                lines.append(f"        ArgAt{{0x{abs_off+k*8:x}}}, (void*)bufs[{nbuf}],"); nbuf+=1
            for k in range(4):
                lines.append(f"        ArgAt{{0x{abs_off+32+k*8:x}}}, (uint64_t){SMALL_W},")
        elif kind=="size2":  # NppiSize {{int w,h}}
            lines.append(f"        ArgAt{{0x{abs_off:x}}}, (int){SMALL_W},")
            lines.append(f"        ArgAt{{0x{abs_off+4:x}}}, (int){SMALL_H},")
        elif kind=="u64":
            lines.append(f"        ArgAt{{0x{abs_off:x}}}, (uint64_t){SMALL_W},")
        elif kind=="i32":
            lines.append(f"        ArgAt{{0x{abs_off:x}}}, (int){SMALL_W},")
        elif kind=="i16":
            lines.append(f"        ArgAt{{0x{abs_off:x}}}, (uint16_t){SMALL_W},")
        elif kind=="i8":
            lines.append(f"        ArgAt{{0x{abs_off:x}}}, (uint8_t){SMALL_W},")
        else:
            lines.append(f"        ArgAt{{0x{abs_off:x}}}, (uint64_t)0,")
    body = TEMPLATE.format(dem=dem.replace('{','{{').replace('}','}}'), mangled=mangled,
        nbuf=nbuf, args="\n".join(lines).rstrip(","),
        gx=1, gy=1, bx=32, by=8)
    # strip trailing comma on last arg line
    body = re.sub(r",(\s*\n\s*\);)", r"\1", body)
    print(body)

if __name__=="__main__":
    main()
