"""Auto-generate x86 host driver (host_cpu.cpp) for lifted CUDA kernels.

Given a KernelSignature, generates C++ that:
  1. Allocates buffers for pointer args (deterministic init)
  2. Sets scalar args to reasonable defaults
  3. Calls the lifted kernel via launchKernel()
  4. Validates output (deterministic check + optional reference)
"""

import re
from pathlib import Path
from typing import List, Set

from utils.kernel_sig import KernelSignature, KernelParam, get_struct_fields

# BUG-043 fix: per-buffer prime multipliers for differentiated initialization.
# Each buffer gets a different pattern so coordinate arrays, weight arrays,
# index arrays, etc. are never identical.  Primes chosen to minimize overlap
# within a 256-element cycle.
_BUF_PRIMES = [3, 7, 11, 17, 23, 31, 37, 43, 53, 61, 71, 79, 89, 97]
_BUF_OFFSETS = [1, 5, 13, 29, 41, 59, 67, 83, 101, 107, 113, 127, 131, 139]


def detect_c4_offsets(ll_path) -> Set[int]:
    """Scan lifted .ll for const_mem bank 4 accesses.

    Returns set of byte offsets used in c[4].  The lifted IR encodes
    constant-bank accesses as GEP into @"const_mem" with indices
    [0, bank, offset].  We look for bank==4.

    BUG-039: c[4] holds 64-bit pointers to __device__ variables.

    Slots stored by the lifter-emitted @__slifter_nv_const4_init ctor
    already hold correct .nv.global addresses at program start; they are
    excluded so the driver's zero-buffer placeholders don't clobber them.
    """
    offsets: Set[int] = set()
    try:
        text = Path(ll_path).read_text()
    except Exception:
        return offsets
    ctor_filled: Set[int] = set()
    m = re.search(r'^define internal void @"__slifter_nv_const4_init"\(\).*?^}',
                  text, re.MULTILINE | re.DOTALL)
    if m:
        for cm in re.finditer(r'@"const_mem", i64 0, i64 4, i64 (\d+)', m.group(0)):
            ctor_filled.add(int(cm.group(1)))
    # Match: getelementptr ... @"const_mem", i64 0, i64 4, i64 N
    for m in re.finditer(r'@"const_mem".*?i64 0, i64 4, i64 (\d+)', text):
        offsets.add(int(m.group(1)))
    return offsets - ctor_filled


def detect_param_ptr_slots(ll_path, kernel_name: str, windows):
    """Classify c[0] param offsets inside `windows`: pointers, i64 scalars,
    or pointer-to-pointer fields.

    For by-value struct params (persist_bp_args, params_t, TensorRef ...),
    interior device-pointer fields live at const-bank-0 offsets inside the
    param window; the auto-driver fills them with scalar patterns, so the
    kernel dereferences garbage and SIGSEGVs. This backward-traces every
    inttoptr operand in `kernel_name`'s body to const_mem bank-0 loads at
    constant offsets. A load only counts when it reaches an inttoptr
    through unscaled arithmetic (passing through mul/shl means the value
    is index/stride math, not a base pointer); both direct i64 loads and
    lo/hi i32 pairs combined via shl-32 are recognized.

    Candidate slots are collected PER inttoptr root. A slot is a POINTER
    only if some root reaches it as the SOLE window-slot candidate — a base
    pointer is always used unaccompanied at least once (the iteration-0
    address). Slots that only ever appear summed with other candidates are
    i64 increments (CUTLASS iterator Params inc_strided_/inc_next_/
    inc_advance_); the old global-union rule filled those with host
    pointers, so the kernel computed ptr+ptr (hi 0x7fff..+0x7fff.. wraps
    negative) and SIGSEGVed (00_basic_gemm, 02_dump_reg_shmem).

    Separately, every other `load i64` from a window offset is an i64
    SCALAR field (CUTLASS TensorRef strides are int64). The wordwise i32
    fill writes {N, N} there — a stride of N|N<<32 — so address math
    explodes (14_ampere_tf32 reference-Gemm SIGSEGV). Those slots must be
    overwritten with a single int64 N. EXCEPT vectorized pairs: a wide
    load whose lo/hi halves meet as operands of one i32 instruction is a
    single ULDC.64 over two ADJACENT i32 fields, not an i64 (cudnn
    bn_fw_tr_1C11_kernel_NCHW loads two desc dims at 0x238/0x23c with one
    ULDC.64 and multiplies them into the element count; the int64 rewrite
    zeroed the hi dim, count = 0, mean = 0/0 = NaN in every stats buffer).
    Such pair slots keep the wordwise fill.

    Third class: a slot whose loaded value is dereferenced AGAIN (the root
    trace passes THROUGH a global load) is a PTR-PTR field — CUTLASS
    LinearCombination alpha_ptr_array/beta_ptr_array are `const* const*`
    with a documented fallback chain (ptr_array -> ptr -> scalar). Filling
    a data buffer there gets its first 8 bytes inttoptr'd (rax shows the
    float fill pattern 0x3ff70a3e_3fe66666) and SIGSEGVs; these slots must
    be zero so the kernel takes the scalar fallback.

    Fourth class: an INDEX-BUFFER slot — a real pointer slot whose LOADED
    values feed ANOTHER inttoptr root's address arithmetic
    (52_hopper_gather_scatter scatter_kernel: `dst + indices[k]*16`).
    Scaling is expected on that path (it is exactly why the unscaled root
    trace never reaches these loads), so each root is also walked through
    ALL arithmetic; loads encountered whose own address is not a
    const-window slot are resolved back (unscaled) to the pointer slot
    they read through. Such side buffers must be filled with ints in
    [0,N), not float patterns — a float fill read as i32 is ~1e9, so the
    scatter lands ~17GB out of bounds.

    Fifth class: GRID-BARRIER EXPECTED-COUNT slots. cudnn semi-persistent
    kernels (bn_bw_1C11_singleread*) retire CTAs through a workspace
    counter: every CTA atomically adds 1, except the arrival that brings
    the count to an expected total, which adds -expected-0x7fffffff,
    flipping the counter's sign bit; all CTAs poll the counter until the
    sign flips. The expected total is an i32 field of the by-value args
    struct (so.69 bn_bw_1C11_args: c0[0x3a4]); any fill other than the
    LAUNCHED gridDim makes the flip unreachable and every thread polls
    forever (singleread_spec: 256 threads at 100% CPU; at -O2 LICM hoists
    the non-atomic poll load and the loop compiles to a literal
    `jmp self`). Detection anchors on the flip constant: an `add` with
    one operand (a zext of) -2147483647 — the other operand back-traces
    through the two's-complement negation (xor -1 / add 1 / and
    0xffffffff) and PHIs to the expected-count load.

    Args:
        ll_path: lifted .ll file (may contain many functions).
        kernel_name: exact lifted symbol whose body to scan.
        windows: [(start, end)] byte ranges from EIATTR_KPARAM_INFO; only
            offsets inside one of them are returned.
    Returns (ptr_slots, i64_scalar_slots, ptrptr_slots, idx_slots,
        count_slots): sorted byte offsets, mutually disjoint (index
        buffers are still pointers, but they are returned only in
        idx_slots so the driver applies the int fill; count slots outrank
        every other class because any other fill deadlocks the kernel).
    """
    try:
        text = Path(ll_path).read_text()
    except Exception:
        return [], [], [], [], []
    # Isolate the target function's body.
    m = re.search(r'^define[^\n]*@"?%s"?\(' % re.escape(kernel_name), text,
                  re.MULTILINE)
    if not m:
        return [], [], [], [], []
    end = text.find('\n}', m.start())
    body = text[m.start():end if end != -1 else len(text)]

    name_re = re.compile(r'^\s*%"([^"]+)" = (\w+)')
    op_re = re.compile(r'%"([^"]+)"')
    gep_re = re.compile(r'@"const_mem", i64 0, i64 0, i64 (\d+)')

    defs = {}       # ssa name -> (op, [operand names], line)
    gep_off = {}    # const_mem bank-0 GEP -> byte offset
    load_src = {}   # load -> (pointer ssa name, is_wide_i64)
    roots = []      # inttoptr source names

    for line in body.splitlines():
        m = name_re.match(line)
        if not m:
            continue
        name, op = m.group(1), m.group(2)
        ops = op_re.findall(line)[1:]  # skip the result name
        defs[name] = (op, ops, line)
        if op == "getelementptr":
            g = gep_re.search(line)
            if g:
                gep_off[name] = int(g.group(1))
        elif op == "load":
            wide = bool(re.match(r'\s*%"[^"]+" = load i64', line))
            load_src[name] = (ops[0] if ops else None, wide)
        elif op == "inttoptr" and ops:
            roots.append(ops[0])

    def resolve_gep(n, depth=0):
        # Loads go through a bitcast (const_mem is an i8 array, so typed
        # loads are `gep -> bitcast -> load`); follow the cast chain.
        while depth < 8:
            if n in gep_off:
                return gep_off[n]
            d = defs.get(n)
            if d and d[0] in ("bitcast", "getelementptr") and d[1]:
                n, depth = d[1][0], depth + 1
                continue
            return None
        return None

    in_win = lambda o: any(s <= o and o + 8 <= e for s, e in windows)

    def trace(start, inner_loads=None):
        """Window-slot offsets reachable from `start` through unscaled
        arithmetic: (lo_i32, hi_i32, wide_i64) sets. When `inner_loads`
        is given, loads from NON-const-window addresses encountered on
        the path record their address SSA name there (ptr-ptr evidence)."""
        lo, hi_s, wide_s = set(), set(), set()
        seen, stack = set(), [(start, False)]  # (name, hi)
        while stack:
            n, hi = stack.pop()
            if (n, hi) in seen:
                continue
            seen.add((n, hi))
            if n in load_src:
                src, wide = load_src[n]
                off = None if src is None else resolve_gep(src)
                if off is not None:
                    if wide:
                        wide_s.add(off)
                    else:
                        (hi_s if hi else lo).add(off)
                elif src is not None and inner_loads is not None:
                    # The traced address came THROUGH a load whose own
                    # address is not a const-window slot: the loaded value
                    # is itself a pointer. Slots feeding THIS load's
                    # address are ptr-ptr fields.
                    inner_loads.add(src)
                continue  # memory boundary — don't trace through loads
            op, ops, line = defs.get(n, (None, (), ""))
            if op is None or op in ("icmp", "fcmp", "call"):
                continue
            if op in ("mul", "udiv", "sdiv", "urem", "srem"):
                continue  # scaled = index math, not a base pointer
            if op == "and" and not re.search(r',\s*4294967295\s*$', line):
                continue  # masking — except the lo-half zext of a 64-bit combine
            if op == "shl":
                if re.search(r',\s*32\s*$', line):
                    stack.extend((c, True) for c in ops)  # hi-half combine
                continue  # other shifts scale
            stack.extend((c, hi) for c in ops)
        return lo, hi_s, wide_s

    def cand_set(lo, hi_s, wide_s):
        cand = ({o for o in lo if o % 8 == 0 and (o + 4) in hi_s}
                | {o for o in wide_s if o % 8 == 0})
        return {o for o in cand if in_win(o)}

    # Per-root candidate sets; the sole-candidate rule picks true base
    # pointers, companion-only slots are i64 increments (see docstring).
    # Sole evidence is two-tier: a root whose trace crossed an inner
    # (non-const-window) load gives only WEAK sole status — the true base
    # may be the loaded value (a spilled iterator pointer rematerialized
    # from local memory) with the window slot as added offset material.
    # A weak-sole slot that ALSO rides along in a multi-candidate set
    # next to a strong-sole base pointer is a companion increment, not a
    # pointer (94_ada_fp8_blockwise: inc_strided_ at 0x238 was sole only
    # for a spill-reload root, and accompanied strong base 0x250
    # elsewhere — pointer-filling it made the kernel compute ptr+ptr and
    # SIGSEGV at hi=0x7fff+0x7fff). Weak-sole alone does NOT demote:
    # 29_3xtf32_complex_gemm's true pointer 0x260 is reached only through
    # spill round-trips but never accompanies another base.
    appeared, inner_loads = set(), set()
    per_root = []
    for r in roots:
        il = set()
        cand = cand_set(*trace(r, il))
        inner_loads |= il
        appeared |= cand
        per_root.append((cand, bool(il)))

    # SLIFTER_PSLOT_DEBUG=1: emit per-root provenance records (slot
    # constraint forensics — the analysis behind temp/params_byvalue).
    import os as _os
    if _os.environ.get("SLIFTER_PSLOT_DEBUG"):
        for _i, ((cand, has_il), _r) in enumerate(zip(per_root, roots)):
            print(f"[pslot-debug] root[{_i}] {_r}: "
                  f"cand={sorted(hex(c) for c in cand)} inner={int(has_il)}",
                  flush=True)

    strong, weak = set(), set()
    for cand, has_il in per_root:
        if len(cand) == 1:
            (weak if has_il else strong).update(cand)
    weak -= strong
    demoted = set()
    for cand, _has_il in per_root:
        if len(cand) > 1 and cand & strong:
            demoted |= cand & weak
    sole = strong | (weak - demoted)

    # Double-deref slots (ptr-ptr) outrank the sole-candidate rule: the
    # inner load's address is built from window slots that must be zero.
    # The SOLE-candidate discipline applies to inner traces exactly as it
    # does to roots: a real inner pointer's address resolves to ONE window
    # slot (alpha_ptr_array, conv Params 0x258/0x270, scatter indices).
    # Multi-candidate inner traces are phi soup — SASS register reuse
    # merges loaded DATA back into address phis, so the back-trace fans
    # out over unrelated slots (so.75 bn_fw_tr_1C11_singleread_spec:
    # {0x278,0x2e8,0x2f0,0x340,0x350,0x358} from one trace). Pointer-
    # filling those clobbered i32 scalar fields — 0x358/0x35c took heap
    # pointer halves and the derived element index walked x[] out of the
    # map (ASLR-flaky SIGSEGV). Non-sole inner candidates keep the
    # wordwise {N, N} struct fill.
    ptrptr = set()
    for src in inner_loads:
        cand = cand_set(*trace(src))
        if len(cand) == 1:
            ptrptr |= cand
        elif cand and _os.environ.get("SLIFTER_PSLOT_DEBUG"):
            print(f"[pslot-debug] multi-cand inner trace (not ptr-ptr): "
                  f"{sorted(hex(c) for c in cand)}", flush=True)
    sole -= ptrptr
    appeared |= ptrptr

    # CREDIBLE address-path evidence, for the pair/dim demotion vetoes
    # below. Candidates from roots whose trace crossed an inner load are
    # phi soup (register reuse merges loaded DATA into address phis) —
    # the same evidence tier the sole rule already discounts to "weak"
    # and the ptr-ptr rule rejects outright when multi-candidate. The
    # blanket `appeared` veto let exactly that soup pin cudnn_cnn so.61
    # SbarGenBitMaskFwdGeneral 0x220 as i64: its only address-path
    # appearances are inner=1 multi-candidate traces, while its hi half
    # is the DIVISOR of the linearized-blockIdx magic-divide (udiv after
    # IntDivRecognize) plus an icmp-vs-runtime tile guard. The int64-N
    # fill zeroed the divisor word -> quotient 0xffffffff -> wild index
    # SIGSEGV. Companion-increment protection (94_ada inc_strided_)
    # is preserved: base+increment address arithmetic reaches the
    # inttoptr add chain directly, with no load crossing (inner=0).
    appeared_strong = ptrptr.copy()
    for cand, has_il in per_root:
        if not has_il:
            appeared_strong |= cand

    # Fourth class: index buffers (see docstring). Walk each root through
    # ALL arithmetic — memory boundaries still stop the walk — collecting
    # loads whose own address is not a const-window slot; the pointer
    # slots those loads read through supply address material, i.e. index
    # arrays. The base-pointer load of the root itself resolves to a
    # window slot and is skipped, so a root never flags its own slot.
    def deep_loads(start):
        out, seen, stack = set(), set(), [start]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in load_src:
                src, _ = load_src[n]
                if src is not None and resolve_gep(src) is None:
                    out.add(src)
                continue
            op, ops, _ln = defs.get(n, (None, (), ""))
            if op is None or op in ("icmp", "fcmp", "call"):
                continue
            stack.extend(ops)
        return out

    idx = set()
    for r in roots:
        for src in deep_loads(r):
            idx |= cand_set(*trace(src)) & sole

    # Companion-only candidates are i64 increments; plus any other wide
    # (i64) load from a window offset is an i64 scalar field.
    all_wide = set(appeared - sole)
    wide_offs = {}  # wide window-load ssa name -> byte offset
    for name, (src, wide) in load_src.items():
        if not wide or src is None:
            continue
        off = resolve_gep(src)
        if off is not None and off % 8 == 0 and in_win(off):
            all_wide.add(off)
            wide_offs[name] = off

    # Pair demotion: a wide window load is NOT an i64 scalar when it is a
    # VECTORIZED PAIR of i32 fields (one ULDC.64 over two adjacent struct
    # ints). The lifter always splits a wide load into trunc (lo) and
    # lshr-32/trunc (hi) halves, so the split idiom itself proves nothing
    # — true TensorRef strides split identically. The structural tell is
    # the halves MEETING as operands of one instruction (cudnn
    # bn_fw_tr_1C11_kernel_NCHW multiplies desc dims: `mul i32 lo, hi` =
    # element count): no use mixes the lo and hi words of a genuine i64
    # at i32 width. Misclassified pairs got the single-int64 N rewrite,
    # zeroing the hi field — count = N*0 = 0, so every loop was 0-trip
    # and mean = 0/0 poisoned the stats buffers with NaN (so.75, 72
    # kernels). Demoted slots keep the wordwise {N, N} struct fill.
    # CREDIBLE address-path evidence (`appeared_strong`) outranks
    # demotion: slots reaching an inttoptr root combined hi<<32|lo ARE
    # 64-bit material. Phi-soup (inner=1) appearances do not veto.
    half_of = {}  # half ssa name -> (wide load name, is_hi)
    for n, (op2, ops2, ln2) in defs.items():
        if not ops2 or ops2[0] not in wide_offs:
            continue
        if op2 == "trunc":
            half_of[n] = (ops2[0], False)
        elif op2 == "lshr" and re.search(r',\s*32\s*$', ln2):
            half_of[n] = (ops2[0], True)
    for n, (op2, ops2, ln2) in defs.items():
        if op2 == "trunc" and ops2 and ops2[0] in half_of \
                and half_of[ops2[0]][1]:
            half_of[n] = half_of[ops2[0]]
    pair_offs = set()
    for _op2, ops2, _ln2 in defs.values():
        sides = {}
        for o2 in ops2:
            if o2 in half_of:
                ld, is_hi = half_of[o2]
                sides.setdefault(ld, set()).add(is_hi)
        pair_offs |= {wide_offs[ld] for ld, s2 in sides.items()
                      if len(s2) == 2}
    pair_offs -= appeared_strong
    if _os.environ.get("SLIFTER_PSLOT_DEBUG") and pair_offs:
        print(f"[pslot-debug] i32-pair slots (kept wordwise, not i64): "
              f"{sorted(hex(c) for c in pair_offs)}", flush=True)

    # icmp discriminator: a HALF of a wide window load used as an i32
    # operand of an icmp against another RUNTIME value is a loop/tile
    # boundary check — a DIMENSION field, not half of a genuine i64.
    # CUTLASS GemmCoord problem_size {m@+0,n@+4,k@+8}: {m,n} arrive as one
    # ULDC.64, the slot classified i64, and the int64-N rewrite wrote
    # {m=N, n=0} — every `coord >= n` tile guard then failed and the
    # kernel exited writing ZERO bytes, so the suite's smoke "pass" was
    # vacuous (05_batched_gemm / 33_ampere_3xtf32_tensorop_symm value
    # audit). The halves never meet in one instruction (m guards row
    # loops, n guards column loops), so the pair rule above cannot see
    # them; the icmp itself is the dimension evidence. Two exclusions
    # keep genuine i64s safe:
    #   1. compares against CONSTANTS prove nothing — an i64's hi half
    #      alone legitimately carries `i64 < 0` / `== 0` range tests;
    #   2. the lifted ISETP.EX 64-bit split compare DOES compare both
    #      halves of a true i64 at i32 width. Its structural signature
    #      (lift_isetp): `select(icmp-eq(hi_a,hi_b), p_low, icmp(hi_a,
    #      hi_b))` — an icmp-eq select condition whose other compare
    #      shares the eq's operand pair. Both those icmps AND every icmp
    #      reachable from the lo-predicate arm (through i1 combiner/
    #      carrier ops) are excluded from dimension evidence.
    icmps = {n: (ops2, ln2) for n, (op2, ops2, ln2) in defs.items()
             if op2 == "icmp"}
    ex_tainted = set()
    for _n, (op2, ops2, _ln2) in defs.items():
        if op2 != "select" or len(ops2) != 3:
            continue
        cond, arm_t, arm_f = ops2
        if cond not in icmps or " = icmp eq " not in defs[cond][2]:
            continue
        cond_ops = set(icmps[cond][0])
        hi_arms = [a for a in (arm_t, arm_f)
                   if a in icmps and set(icmps[a][0]) == cond_ops]
        if not hi_arms:
            continue
        ex_tainted.add(cond)
        ex_tainted.update(hi_arms)
        # the remaining arm is the lo-word predicate chain
        walk = [a for a in (arm_t, arm_f) if a not in hi_arms]
        seen_x = set()
        while walk:
            w = walk.pop()
            if w in seen_x:
                continue
            seen_x.add(w)
            opw, opsw, _lw = defs.get(w, (None, (), ""))
            if opw == "icmp":
                ex_tainted.add(w)
            elif opw in ("and", "or", "xor", "select", "phi", "trunc",
                         "zext", "sext"):
                walk.extend(opsw)
    dim_offs = set()
    for n, (ops2, ln2) in icmps.items():
        if n in ex_tainted:
            continue
        if not re.match(r'\s*%"[^"]+" = icmp \w+ i32 ', ln2):
            continue
        if len(ops2) < 2:
            continue  # constant comparand — see exclusion 1
        for o2 in ops2:
            if o2 in half_of:
                dim_offs.add(wide_offs[half_of[o2][0]])
    # Credible address-path evidence outranks here exactly as it does for
    # the pair rule: a slot combined hi<<32|lo into an inttoptr root is
    # 64-bit material no matter what its halves are compared against.
    # Phi-soup (inner=1 multi-candidate) appearances do NOT veto — see
    # appeared_strong (cudnn_cnn so.61 0x220: hi half is the blockIdx
    # decomposition divisor; the blanket veto kept it i64 and the int64-N
    # fill zeroed the divisor -> div-by-0 -> wild index SIGSEGV).
    dim_offs -= appeared_strong
    if _os.environ.get("SLIFTER_PSLOT_DEBUG") and dim_offs:
        print(f"[pslot-debug] icmp-fed dimension slots (kept wordwise, "
              f"not i64): {sorted(hex(c) for c in dim_offs)}", flush=True)
    pair_offs |= dim_offs

    # Fifth class: grid-barrier expected-count slots (see docstring).
    # Anchor: an `add` with a (zext of) -2147483647 operand; walk the
    # other operand back through negation/PHIs to window loads. mul/div
    # stay barriers — a count materialized as a PRODUCT of fields would
    # need its own fill rule, and tagging the factors would be wrong.
    flip_consts = {n for n, (op2, _o2, ln2) in defs.items()
                   if op2 in ("zext", "sext") and "-2147483647" in ln2}
    in_win4 = lambda o: any(s <= o and o + 4 <= e for s, e in windows)
    count = set()
    for n, (op2, ops2, ln2) in defs.items():
        if op2 != "add":
            continue
        if "-2147483647" not in ln2 and not any(o2 in flip_consts
                                                for o2 in ops2):
            continue
        walk = [o2 for o2 in ops2 if o2 not in flip_consts]
        seen_w = set()
        while walk:
            w = walk.pop()
            if w in seen_w:
                continue
            seen_w.add(w)
            if w in load_src:
                src, wide = load_src[w]
                off = None if src is None else resolve_gep(src)
                if off is not None and not wide and off % 4 == 0 \
                        and in_win4(off):
                    count.add(off)
                continue
            opw, opsw, _lw = defs.get(w, (None, (), ""))
            if opw in ("zext", "sext", "trunc", "xor", "add", "sub",
                       "and", "or", "phi", "bitcast", "select"):
                walk.extend(opsw)
    if _os.environ.get("SLIFTER_PSLOT_DEBUG") and count:
        print(f"[pslot-debug] grid-barrier count slots: "
              f"{sorted(hex(c) for c in count)}", flush=True)

    # Count slots outrank every other class: any other fill deadlocks.
    ptr_slots = sorted(sole - idx - count)
    i64_slots = sorted(all_wide - sole - ptrptr - count - pair_offs)
    return ptr_slots, i64_slots, sorted(ptrptr - count), \
        sorted(idx - count), sorted(count)


def _generate_c4_extern_decl() -> str:
    """Generate the file-scope extern declaration for const_mem (BUG-039).

    Must be placed at file scope (before main) because C++ does not allow
    'extern "C"' with 'alignas' and 'thread_local' at block scope.
    """
    return ('// BUG-039: const_mem extern for c[4] device variable population\n'
            'extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];')


def _generate_c4_population(c4_offsets: Set[int]) -> str:
    """Generate C++ code to allocate buffers and store pointers in const_mem[4].

    Each c[4] slot gets an 8-byte pointer to a 4KB zero-initialized heap buffer.
    This is a generic approach: the kernel will read the pointer, then dereference
    it to access the device variable.  Zero-initialized buffers won't give
    numerically correct results, but they prevent SIGSEGV crashes.

    BUG-039: auto-populate c[4] device variable pointers for hecbench kernels.
    The extern declaration for const_mem must be at file scope (see
    _generate_c4_extern_decl); this function generates only the population
    code that goes inside main().
    """
    lines = []
    lines.append('    // BUG-039: Auto-populate c[4] device variable pointers')
    for i, offset in enumerate(sorted(c4_offsets)):
        # Allocate a 4KB buffer for each c[4] slot (generous default)
        lines.append(f'    uint8_t* c4_buf{i} = new uint8_t[4096]{{}};')
        lines.append(f'    uint64_t c4_addr{i} = (uint64_t)(uintptr_t)c4_buf{i};')
        lines.append(f'    memcpy(&const_mem[4][{offset}], &c4_addr{i}, sizeof(uint64_t));')
    return '\n'.join(lines)


def generate_host(sig: KernelSignature, sm: int = 90, sm_mode: str = "normal",
                  c4_offsets: Set[int] = None,
                  ref_match=None,
                  ref_header: str = None,
                  init_hints: dict = None,
                  extra_ptr_slots: List[int] = None,
                  extra_i64_slots: List[int] = None,
                  extra_ptrptr_slots: List[int] = None,
                  extra_idx_slots: List[int] = None,
                  extra_count_slots: List[int] = None,
                  ctaid_y: bool = None) -> str:
    """Generate host_cpu.cpp content for a single kernel.

    Args:
        sig: Kernel signature extracted from reference LLVM IR.
        sm: Target SM architecture (e.g. 90).
        sm_mode: Size mode for buffers/grid.
            "normal" — N=32, BUF_SIZE=65536, blockDim=256 (default, safe for compiled-in constants)
            "small"  — N=1, BUF_SIZE=64, blockDim=1 (fast single-thread x86 smoke test)
        c4_offsets: Set of byte offsets into const_mem bank 4 that the kernel
            accesses.  If non-empty, the generated driver will allocate heap
            buffers and store their addresses at these offsets (BUG-039).
        ref_match: Optional `utils.reference_match.RefMatch` describing how to
            invoke the bench's reference.h CPU implementation. When provided,
            the driver emits a strict element-wise comparison (memcmp for
            integer outputs, tolerance compare for floating-point outputs)
            between the kernel's output buffer and the reference output.
            When None, the driver falls back to the smoke check.
        ref_header: Path (project-relative or absolute) to reference.h to
            #include. Required when ref_match is provided.
        init_hints: Optional per-test input-domain overrides (loaded from the
            bench's `driver_hints.json`). The default [1.0, 3.56] buffer init
            protects log/rcp/rsqrt kernels but is domain-INVALID for kernels
            needing small inputs (lci-cuda's spectral sums NaN on it even in
            a host-C++ reference run of the same math). Shape:
              {"buffers": {"<buf_ordinal>": {"expr": "<C++ expr of i>",
                                             "first": "<C++ literal>"}},
               "scalars": {"<scalar_ordinal>": "<C++ literal>"},
               "int_scalars": {"<int_scalar_ordinal>": "<C++ literal>"},
               "blockDim": <int>,
               "gridDim": <int or "N">,
               "pslot_fills": {"<hex c0 offset, e.g. 0x240>": "zero"},
               "word_fills": {"<hex c0 offset>": <u32 int>},
               "idx_mod": {"<hex c0 offset of idx slot>": <int modulus>}}
            "scalars" ordinals count float/double scalars; "int_scalars"
            ordinals count int/int64 scalars (both in parameter order).
            Needed for kernels whose int params are dims/strides counts the
            gather math depends on (e.g. tensorT's dim_input — N=32 there
            walks 32 garbage strides and segfaults).
            "gridDim" pins the launch grid for kernels whose outputs are
            indexed by blockIdx.x (one CTA per output row). The default
            ceil(N/blockDim) collapses to 1 block at N=32: the kernel
            writes out[0] only and the strict reference compare flags the
            never-written tail as wrong_result (wedford: the app launches
            grid(feature_size); 31/32 "differ" were prefill vs reference).
            The string "N" tracks the scalar problem size, so small mode
            (N=1) degrades consistently to a single block.
            "pslot_fills" overrides the synthesized fill of a detected
            slot's buffer by its absolute c[0] offset, for ANY slot class
            (extra_ptr_slots, extra_ptrptr_slots, extra_idx_slots — slot
            classification varies run-to-run with lift nondeterminism, so
            the hint must reach the slot wherever it lands).
            "zero" is for lock/semaphore workspaces the real host
            cudaMemsets to 0 — e.g. CUTLASS split-K `params.semaphore`
            (58_ada_fp8_gemm, 36_gather_scatter_fusion: Semaphore::wait()
            spins until *lock == 0; every float-pattern element — and
            every synthesized ppslot POINTER — is a nonzero i32 bit
            pattern, so the default fills spin forever). On a ptrptr/idx
            slot "zero" drops the synthesized indirection / index pattern
            and allocates one zeroed buffer instead: lock words read
            through the slot must be 0, which no valid-pointer fill can
            satisfy.
            "idx" declares the slot an index-array pointer the idx scan
            missed (epilogue-side scatter chains evade deep_loads):
            int32 values in [0,N) like a native idx slot. 36's
            ptr_scatter_D_indices at 0x378 classified ptrptr; its
            synthesized pointers read as i32 row indices ~4e8 walked the
            D scatter ~GBs out of bounds (post-semaphore movaps SIGSEGV).
            "word_fills" pins 4-byte scalar fields INSIDE by-value struct
            param windows by absolute c[0] offset, written after the struct
            filler and pslot writes (so they win). For constraint-coupled
            dims the generic filler cannot know — e.g. CUTLASS sparse
            problem_size {m,n,k}: the {m,n} pair is loaded by one LDC.64
            (classified i64 → filler writes {m=32,n=0}) and 2:4-sparse A
            needs k % 64 == 0 or the kSparse rows leave an 8-byte residue
            and a 16B movaps faults (15_ampere_sparse class). Values may
            also be {"f32": <num>} (FLOAT field — bit pattern computed
            here; cudnn singleread invCount/fracUpscale/threshold) or
            {"expr": "<C++>"} (launch-coupled scalar computed by the
            driver from blockDim/gridDim — perThreadIters).
            "buf_init" pins a plain side DATA buffer at an absolute c[0]
            offset, AUTHORITATIVE over the slot-scan class (the cudnn
            singleread x/dy/scale fields classify ptr-ptr; their pointer
            staggering read as f32 overflows F2I.S64 into the error-
            sentinel path — bn_oracle (b)). Shape per offset:
              {"fill": "pattern"|"zero", "elem": <optional override>,
               "words": {"<hex byte off>": "<C++ u32 expr>"},
               "check_words": {"<hex byte off>": "<C++ u32 expr>"},
               "check_zero_upto": "<hex byte bound>"}
            "words" pins 32-bit words inside the allocation BEFORE launch
            (the singleread grid-barrier counter pre-bias 0x80000000 -
            blockDim*gridDim); "check_words"/"check_zero_upto" assert the
            post-launch ws census (protocol self-restore: slots+flags
            zeroed, counter re-armed, errStatus 0) and FAIL the row on
            mismatch.
        extra_ptr_slots: absolute c[0] byte offsets (inside by-value struct
            param windows) that hold device pointers, from
            detect_param_ptr_slots. Each gets a fresh side allocation whose
            address is pinned at the exact offset with ArgAt, overriding the
            scalar filler written by the struct decomposition. None = not
            scanned; [] = scanned, nothing found (both emit no slots, but
            the scan-marker comment distinguishes regenerated drivers).
        extra_i64_slots: absolute c[0] byte offsets (inside by-value struct
            param windows) the kernel loads as a single i64 scalar (CUTLASS
            TensorRef strides). The wordwise i32 filler writes {N, N} there
            — a stride of N|N<<32 that explodes address math — so each slot
            is rewritten with one int64 N after the regular args. Same
            None/[] marker semantics as extra_ptr_slots.
        extra_ptrptr_slots: absolute c[0] byte offsets the kernel loads and
            dereferences TWICE (CUTLASS LinearCombination alpha_ptr_array/
            beta_ptr_array are `const* const*` with a documented fallback
            chain ptr_array -> ptr -> scalar). Filled with int64 0 so the
            kernel takes the scalar fallback; any data buffer here gets its
            first 8 bytes inttoptr'd and SIGSEGVs (00_basic_gemm). Same
            None/[] marker semantics as extra_ptr_slots.
        extra_idx_slots: absolute c[0] byte offsets holding INDEX-ARRAY
            pointers — values loaded through them feed another pointer's
            address arithmetic (52_hopper_gather_scatter scatter_kernel
            `dst + indices[k]*16`). Each gets a side buffer like
            extra_ptr_slots but filled with int32 values in [0,N) using
            the bijective (i*prime+off)%N pattern (same convention as the
            int main-param fill); the float pattern read as i32 is ~1e9
            and scatters ~17GB out of bounds. Same None/[] marker
            semantics as extra_ptr_slots.
        extra_count_slots: absolute c[0] byte offsets of GRID-BARRIER
            expected-CTA-count i32 fields (cudnn bn_bw_1C11_singleread*
            retirement counters). Written with the driver's launched
            gridDim — any other value makes the barrier's sign-flip
            unreachable and the kernel polls forever. Same None/[]
            marker semantics as extra_ptr_slots.
        ctaid_y: True when the lifted kernel consumes SR_CTAID.Y (a
            const-bank-0 load at 0x24, the wrapper ABI slot SRSubstitute
            maps it to — run_hecbench._detect_ctaid_y). Such kernels bake
            a 2D launch contract the 1D default can't see: cudnn
            bn_pointwise splits the batch over ctaid.y (each handles rows
            {2y, 2y+1}; the app launches gridY=ceil(N/2)), so a 1D launch
            computed batch rows 0..1 of 32 and the other 30 rows of
            "output" were prefill — vacuous smoke pass (oracle
            footnote 1). When True and no "gridDimY" hint is given, the
            driver pins gridDimY=N (the wedford f8a48591 gridDim.x "one
            CTA per problem row" contract extended to y; over-launching a
            rows-per-CTA-y kernel just touches extra in-bounds rows) and
            launches via launchKernel2D. init_hints {"gridDimY": <int or
            "N">} overrides the pin per test. None/False = 1D launch.
    """
    lines = []
    lines.append('#include <cstdint>')
    lines.append('#include <iostream>')
    lines.append('#include <cmath>')
    lines.append('#include <cstring>')
    if ref_match is not None and (ref_match.header_source or ref_match.func.source):
        # Common headers reference.h often expects to be in scope from
        # the enclosing main.cu — add them so embedded code compiles.
        lines.append('#include <vector>')
        lines.append('#include <algorithm>')
        lines.append('#include <cfloat>')
        lines.append('#include <climits>')
        lines.append('#include <cstdlib>')
    lines.append('#include "kernel_wrapper.h"')
    lines.append('')
    # Inline reference.h directly. We don't `#include "reference.h"` because
    # it often references CUDA-only types (half, __nv_bfloat16) and free
    # functions only defined in main.cu — embedding the body lets us prepend
    # stubs for the troublesome types so unused sibling functions still
    # compile. The matcher already rejects functions whose own signature
    # uses these types, so the matched call site is unaffected.
    if ref_match is not None:
        body = ref_match.header_source or ref_match.func.source
        if body:
            stubs = []
            # Stub CUDA half / bfloat16 if referenced. These appear only in
            # unused sibling functions (the matcher won't match through them).
            if re.search(r"\b__half\b|\bhalf\b", body):
                stubs.append('struct __slf_half_stub { float v; '
                             '__slf_half_stub() : v(0) {} '
                             '__slf_half_stub(float f) : v(f) {} '
                             'operator float() const { return v; } };')
                stubs.append('typedef __slf_half_stub __half;')
                stubs.append('typedef __slf_half_stub half;')
            if re.search(r"\b__nv_bfloat16\b|\bbfloat16\b", body):
                stubs.append('typedef __slf_half_stub __nv_bfloat16;')
                stubs.append('typedef __slf_half_stub bfloat16;')
            for s in stubs:
                lines.append(s)
            if stubs:
                lines.append('')
            lines.append(body)
            lines.append('')
    lines.append(f'extern "C" void {sig.name}();')
    lines.append('')
    # BUG-039: file-scope extern for const_mem if c[4] offsets detected
    if c4_offsets:
        lines.append(_generate_c4_extern_decl())
        lines.append('')
    lines.append('int main() {')
    if sm_mode == "small":
        # BUG-045/058: Timeout kernels on single-threaded x86.  Reduce everything:
        # N=1, BUF_SIZE=1024, blockDim=1 (single thread).  This gives gridDim=1
        # with exactly 1 thread — the absolute minimum for a smoke test.
        # BUF_SIZE=1024 (not 64) because kernels often have compiled-in constant
        # offsets (e.g. LSTM weight offsets up to 0x18c=396 bytes) that would
        # exceed a tiny 64-element buffer.
        lines.append('    constexpr int N = 1;')
        lines.append('    constexpr int BUF_SIZE = 1024;')
    else:
        # BUG-038/058 fix: Kernels often have compiled-in large constants (e.g. NSIZE=2003)
        # or per-thread strides (e.g. LSTM with stride 20000) that index far beyond
        # small buffers.  Use a flat BUF_SIZE of 16M elements so that even kernels with
        # blockDim=256 and stride=20000 don't overflow (256*20000+100 = ~5.12M < 16M),
        # and sparse kernels with N^3 indexing also fit (256*32^3 = ~8.4M < 16M).
        lines.append('    constexpr int N = 32;')
        lines.append('    constexpr int BUF_SIZE = 16777216;')
    if sm_mode == "small":
        lines.append('    constexpr int blockDim = 1;')
    else:
        # Kernels with a workgroup size baked into the SASS (loop strides,
        # __shared__ scratch sized for exactly N threads) need the matching
        # launch blockDim (quicksort: LQSORT_LOCAL_WORKGROUP_SIZE). Hint
        # outranks ELF truth (EIATTR_MAX_THREADS via sig.block_dim — cudnn
        # BD=512 kernels hang at the 256 default: their intra-CTA
        # reduction waits on warps 8-15 that were never launched).
        bd = (init_hints or {}).get("blockDim") or sig.block_dim
        lines.append(f'    constexpr int blockDim = {int(bd) if bd else 256};')
    # Per-CTA-output kernels (out[blockIdx.x] reductions) bake a launch
    # contract the ceil(N/blockDim) default can't see — the app launches one
    # block per output row (wedford: grid(feature_size)). With gridDim=1 only
    # out[0] is ever written and the strict reference compare correctly
    # flags the stale tail. driver_hints {"gridDim": "N"} (or an int) pins
    # the contract; "N" scales with the problem size in every sm_mode.
    gd = (init_hints or {}).get("gridDim")
    if gd:
        gd_expr = 'N' if gd == "N" else str(int(gd))
        lines.append(f'    constexpr int gridDim = {gd_expr};  '
                     '// driver_hints: launch contract')
    else:
        lines.append('    constexpr int gridDim = (N + blockDim - 1) / blockDim;')
    # 2D launch contract: kernels that consume SR_CTAID.Y compute one
    # y-slice of work per CTA-y; with the default 1D launch only slice 0
    # is ever computed and the rest of the "output" is prefill
    # (bn_pointwise: batch rows 0..1 of 32). Pin gridDimY like the
    # gridDim.x wedford contract — one CTA-y per problem row.
    gy = (init_hints or {}).get("gridDimY")
    if gy:
        gy_expr = 'N' if gy == "N" else str(int(gy))
        lines.append(f'    constexpr int gridDimY = {gy_expr};  '
                     '// driver_hints: 2D launch contract')
    elif ctaid_y:
        lines.append('    constexpr int gridDimY = N;  '
                     '// SR_CTAID.Y consumed: 2D launch contract')
    two_d = bool(gy or ctaid_y)
    lines.append('')

    # Allocate buffers and scalars
    buf_idx = 0
    int_count = 0
    launch_args = []
    output_bufs = []  # (var_name, size)
    buffer_elems = {}  # every main pointer buffer, including reference inputs
    scalar_idx = 0    # ordinal of float/double scalars, for init_hints["scalars"]

    for p in sig.params:
        # Authoritative layout: when the ELF's EIATTR_KPARAM_INFO gave us the
        # param's absolute c[0] offset (kernel_sig.apply_elf_arg_info), pin
        # it so writeArg's sizeof-alignment heuristic can never drift from
        # the real ABI (an 8-byte by-value struct guessed as a 4-byte int
        # shifted every later param — aop-cuda SIGSEGV).
        if p.elf_offset >= 0:
            launch_args.append(f'ArgAt{{0x{p.elf_offset:x}}}')
        if p.is_pointer:
            var = f"buf{buf_idx}"
            # BUG-043 fix: each buffer gets a different prime/offset pair so that
            # coordinate arrays, weight arrays, etc. are never identical.  Values
            # are shifted to [1.0, 3.56] range to avoid domain violations (log(0),
            # rcp(0), rsqrt(negative), exp2 overflow).
            prime = _BUF_PRIMES[buf_idx % len(_BUF_PRIMES)]
            offset = _BUF_OFFSETS[buf_idx % len(_BUF_OFFSETS)]
            buf_idx += 1
            # Element type from the demangled ctype via _elem_for_param,
            # shared with generate_host_cuda. The old inline chain here
            # disagreed for __half (float buffers): half
            # tensors were filled with float bit patterns read as 2 packed
            # lanes — values landed in normal range (f16 tops out at
            # 65504, unlike bf16 whose lo half-words reached ~1e38 and
            # Inf'd the so.75 NCHW stats) but were uncontrolled, and the
            # type check expected uint16_t. 16-bit float buffers are
            # uint16_t with exact-in-storage fills below.
            ct = p.ctype.replace('const ', '')
            elem = _elem_for_param(p)
            bf16 = 'bfloat' in ct
            f16 = not bf16 and '__half' in ct
            # BUG-043: Differentiated, domain-safe initialization per buffer.
            # Float/double: values in [1.0, 3.56] — safe for log, rcp, rsqrt, exp2.
            # Integer: values in [0, N) — safe as array indices.
            hint = ((init_hints or {}).get("buffers") or {}).get(str(buf_idx - 1))
            if hint and hint.get("elem"):
                # Pointer-to-struct params (work_record<uint>*) default to a
                # float buffer, but the kernel reads raw integer fields —
                # float bit patterns become ~1e9 indices. Let hints force the
                # element type so exprs can build valid records.
                elem = hint["elem"]
            lines.append(f'    {elem}* {var} = new {elem}[BUF_SIZE];')
            if hint and hint.get("expr"):
                # Per-test domain override (driver_hints.json) — see docstring.
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {var}[i] = {hint["expr"]};  // driver_hints')
                if hint.get("first") is not None:
                    lines.append(f'    {var}[0] = {hint["first"]};  // driver_hints')
            elif bf16 and elem == 'uint16_t':
                # Normal bf16 values in [1.0, 2.0): exponent 0x3F80 with a
                # varied 7-bit mantissa — same spirit as the float fill's
                # [1.0, 3.56] domain (no NaN/Inf/denormal/negative).
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {var}[i] = '
                             f'static_cast<uint16_t>(0x3F80 | ((i * {prime} + {offset}) % 128));')
            elif f16 and elem == 'uint16_t':
                # Normal f16 values in [1.0, 2.0): exponent 0x3C00 with a
                # varied 10-bit mantissa — exact in storage (the
                # so69_oracle audit drivers used the same [1,2) grid;
                # __half2 lanes are simply 2 such scalars).
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {var}[i] = '
                             f'static_cast<uint16_t>(0x3C00 | ((i * {prime} + {offset}) % 1024));')
            elif elem in ("float",):
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {var}[i] = static_cast<float>((i * {prime} + {offset}) % 256) * 0.01f + 1.0f;')
            elif elem in ("double",):
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {var}[i] = static_cast<double>((i * {prime} + {offset}) % 256) * 0.01 + 1.0;')
            else:
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {var}[i] = static_cast<{elem}>((i * {prime} + {offset}) % (N > 0 ? N : 1));')
            if hint and hint.get("ptr_slots"):
                # Struct-of-pointers params (gmm's clusters_t = 8 float*
                # fields): the kernel loads device pointers out of this
                # buffer and dereferences them, so the listed 8-byte slots
                # must hold addresses of real allocations, not data patterns.
                for slot in hint["ptr_slots"]:
                    side = f"{var}_p{slot}"
                    lines.append(f'    float* {side} = new float[1 << 20];')
                    lines.append(f'    for (int i = 0; i < (1 << 20); i++) {side}[i] = static_cast<float>((i * {prime} + {offset} + {slot}) % 256) * 0.01f + 1.0f;')
                    lines.append(f'    reinterpret_cast<int64_t*>({var})[{slot}] = static_cast<int64_t>(reinterpret_cast<intptr_t>({side}));  // driver_hints ptr_slots')
            buffer_elems[var] = elem
            if p.is_output:
                output_bufs.append((var, elem))
            launch_args.append(var)
        elif p.is_byval_struct:
            # BUG-051: Struct by-value params are serialized as individual scalar
            # fields in const_mem.  Decompose into separate writeArg calls.
            fields = get_struct_fields(p.struct_name)
            if fields:
                from utils.kernel_sig import (get_struct_fill_overrides,
                                              KNOWN_STRUCT_FILL_TAG)
                overrides = get_struct_fill_overrides(p.struct_name)
                if overrides:
                    # Record the structural-fill policy in the generated
                    # driver so the layout is inspectable.
                    lines.append(f'    // {p.struct_name}: '
                                 f'{KNOWN_STRUCT_FILL_TAG}')
                for field_idx, (field_ctype, _) in enumerate(fields):
                    if field_idx in overrides:
                        # Structural field needing a domain-valid value
                        # (cudnnTensorStruct nbDims=4 — see
                        # kernel_sig._KNOWN_STRUCT_FILLS).
                        launch_args.append(overrides[field_idx])
                    elif field_ctype == 'float':
                        launch_args.append('2.0f')
                    elif field_ctype == 'double':
                        launch_args.append('2.0')
                    elif field_ctype == 'int64_t':
                        launch_args.append('static_cast<int64_t>(N)')
                    else:
                        # int and promoted small types
                        launch_args.append('N')
            else:
                # Unknown struct: estimate field count from size, assume 4-byte fields
                n_fields = max(1, p.size // 4)
                for _ in range(n_fields):
                    launch_args.append('N')
        elif p.ctype == 'float':
            sh = ((init_hints or {}).get("scalars") or {}).get(str(scalar_idx))
            scalar_idx += 1
            # BUG-043: Use 2.0f instead of 1.0f — safer for kernels where
            # scalar params are divisors, exponents, or scaling factors.
            launch_args.append(sh if sh else '2.0f')
        elif p.ctype == 'double':
            sh = ((init_hints or {}).get("scalars") or {}).get(str(scalar_idx))
            scalar_idx += 1
            launch_args.append(sh if sh else '2.0')
        elif p.ctype == 'bool':
            # BUG-051/BUG-066: bool params are 1 byte in CUDA (ULDC.U8).
            # Must pass as (bool)true so writeArg deduces sizeof(bool)=1.
            # Passing int literal 1 writes 4 bytes → corrupts subsequent
            # param offsets → SIGSEGV.
            launch_args.append('(bool)true')
        elif p.ctype in ('int', 'int32_t', 'unsigned int', 'unsigned', 'uint32_t'):
            # 'j'-mangled params demangle to 'unsigned int'; they are 4-byte
            # writes like int and must share the int_scalars ordinal space —
            # the old bare-'N' fallthrough made hints silently unreachable
            # (split-cuda nbits/startbit).
            ih = ((init_hints or {}).get("int_scalars") or {}).get(str(int_count))
            launch_args.append(ih if ih else 'N')
            int_count += 1
        elif 'long' in p.ctype or p.ctype in ('int64_t', 'uint64_t', 'size_t'):
            # Demangled 64-bit scalars arrive as 'long'/'unsigned long'/
            # 'long long' (mangle 'l'/'m'/'x') — NOT 'int64_t'.  They must be
            # written as 8 bytes: passing bare 'N' (int) makes writeArg emit
            # 4 bytes, shifting every subsequent param offset and feeding the
            # kernel garbage upper halves (nlll-cuda SIGSEGV root cause).
            ih = ((init_hints or {}).get("int_scalars") or {}).get(str(int_count))
            launch_args.append(f'static_cast<int64_t>({ih})' if ih else 'static_cast<int64_t>(N)')
            int_count += 1
        else:
            launch_args.append('N')

    # Kernel storage dtype from the mangled template args (length-prefixed
    # tokens: '6__half', '7__half2', '13__nv_bfloat16', '14__nv_bfloat162').
    # Pointer fields detected INSIDE by-value structs carry no C type of
    # their own, but for half/bf16-templated kernels they are the bulk
    # tensors (bn singleread args x/dy/dx) read as 16-bit lanes — their
    # data fills must be exact-in-storage like the top-level buffer fills,
    # not float bit patterns (so69_oracle footnote 3).
    nm = sig.name or ''
    if re.search(r'\d+__nv_bfloat162?', nm):
        storage16 = ('0x3F80', 128)    # bf16: [1,2) over the 7-bit mantissa
    elif re.search(r'\d+__half2?', nm):
        storage16 = ('0x3C00', 1024)   # f16:  [1,2) over the 10-bit mantissa
    else:
        storage16 = None

    # buf_init: protocol-pinned side buffers at absolute c[0] offsets (see
    # init_hints docstring). The hint is AUTHORITATIVE over the slot scans:
    # a buf_init offset gets its plain data side allocation no matter which
    # class (pslot/ppslot/idx/i64) the scan put it in — the cudnn singleread
    # x/dy/scale fields classify ptr-ptr, and the synthesized pointer
    # staggering read as f32 overflows F2I.S64 into the error-sentinel path
    # (bn_oracle (b)). Slot-scan emissions for these offsets are dropped.
    buf_init = (init_hints or {}).get("buf_init") or {}
    if buf_init:
        _bi_offs = {int(k, 16) for k in buf_init}
        extra_ptr_slots = ([s for s in extra_ptr_slots if s not in _bi_offs]
                           if extra_ptr_slots is not None else None)
        extra_i64_slots = ([s for s in extra_i64_slots if s not in _bi_offs]
                           if extra_i64_slots is not None else None)
        extra_ptrptr_slots = ([s for s in extra_ptrptr_slots
                               if s not in _bi_offs]
                              if extra_ptrptr_slots is not None else None)
        extra_idx_slots = ([s for s in extra_idx_slots if s not in _bi_offs]
                           if extra_idx_slots is not None else None)

    if extra_ptr_slots is not None:
        lines.append(f'    // pslot-scan: {len(extra_ptr_slots)} pointer slot(s) '
                     'detected in by-value struct param windows')
    if extra_ptr_slots:
        # Interior device-pointer fields of by-value struct params: pin a
        # fresh allocation's address at the absolute c[0] offset (written
        # after the regular args, overriding the struct's scalar filler).
        pslot_fills = (init_hints or {}).get("pslot_fills") or {}
        for slot in sorted(extra_ptr_slots):
            side = f'pslot_{slot:x}'
            if pslot_fills.get(f'0x{slot:x}') == 'zero':
                # Per-test hint: lock/semaphore workspace — the real host
                # cudaMemsets it to 0 (see init_hints docstring).
                lines.append(f'    float* {side} = new float[BUF_SIZE]();'
                             '  // driver_hints pslot_fills: zero')
            elif pslot_fills.get(f'0x{slot:x}') == 'idx':
                # Per-test hint: index-array the idx scan missed.
                lines.append(f'    int32_t* {side} = new int32_t[BUF_SIZE];'
                             '  // driver_hints pslot_fills: idx')
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {side}[i] = '
                             f'(i * 13 + {slot & 0xff}) % (N > 0 ? N : 1);')
            elif storage16:
                base, mod = storage16
                lines.append(f'    uint16_t* {side} = new uint16_t[BUF_SIZE];')
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {side}[i] = '
                             f'static_cast<uint16_t>({base} | ((i * 13 + {slot & 0xff}) % {mod}));')
            else:
                lines.append(f'    float* {side} = new float[BUF_SIZE];')
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {side}[i] = '
                             f'static_cast<float>((i * 13 + {slot & 0xff}) % 256) * 0.01f + 1.0f;')
            launch_args.append(f'ArgAt{{0x{slot:x}}}')
            launch_args.append(side)

    if extra_i64_slots is not None:
        # "(pair-aware, icmp-dim, strong-veto)" marks drivers generated
        # after the i32-pair demotion, the icmp dimension discriminator,
        # AND the credible-evidence demotion veto (appeared_strong) in
        # detect_param_ptr_slots. The marker makes the inference policy
        # inspectable in each freshly generated driver.
        lines.append(f'    // i64-scan: {len(extra_i64_slots)} i64 scalar slot(s) '
                     'detected in by-value struct param windows '
                     '(pair-aware, icmp-dim, strong-veto)')
    if extra_i64_slots:
        for slot in sorted(extra_i64_slots):
            launch_args.append(f'ArgAt{{0x{slot:x}}}')
            launch_args.append('static_cast<int64_t>(N)')

    if extra_ptrptr_slots is not None:
        # "(sole-trace)" records that ptr-ptr evidence is restricted to
        # sole-candidate inner traces.
        lines.append(f'    // pp-scan: {len(extra_ptrptr_slots)} ptr-ptr slot(s) '
                     'detected in by-value struct param windows (sole-trace)')
    if extra_ptrptr_slots:
        # Double-deref slots: the kernel loads a POINTER from this slot's
        # buffer and dereferences it. A 0 scalar nulls slots the pslot scan
        # had proven are real pointer fields (conv Params 0x258/0x270:
        # 17/26/30 SIGSEGV), and a plain float fill makes the inner deref a
        # wild pointer. Synthesize the indirection: an array of pointers
        # staggered (16B-aligned) into one over-allocated target buffer —
        # valid memory whether the kernel treats the slot as plain data
        # (pointer bits read as garbage values) or double-derefs.
        pslot_fills = (init_hints or {}).get("pslot_fills") or {}
        for slot in sorted(extra_ptrptr_slots):
            side = f'ppslot_{slot:x}'
            if pslot_fills.get(f'0x{slot:x}') == 'zero':
                # Per-test hint: lock/semaphore workspace — the real host
                # cudaMemsets it to 0 (36_gather_scatter_fusion: the kernel
                # reads lock WORDS through this slot's pointer; the
                # synthesized pointer staggering reads as nonzero = lock
                # held, and Semaphore::wait() spins forever).
                lines.append(f'    int64_t* {side} = new int64_t[BUF_SIZE]();'
                             '  // driver_hints pslot_fills: zero')
                launch_args.append(f'ArgAt{{0x{slot:x}}}')
                launch_args.append(side)
                continue
            if pslot_fills.get(f'0x{slot:x}') == 'idx':
                # Per-test hint: index-array the idx scan missed (epilogue
                # scatter chains evade deep_loads — 36's 0x378
                # ptr_scatter_D_indices; see init_hints docstring).
                lines.append(f'    int32_t* {side} = new int32_t[BUF_SIZE];'
                             '  // driver_hints pslot_fills: idx')
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {side}[i] = '
                             f'(i * 13 + {slot & 0xff}) % (N > 0 ? N : 1);')
                launch_args.append(f'ArgAt{{0x{slot:x}}}')
                launch_args.append(side)
                continue
            lines.append(f'    float* {side}_tgt = new float[BUF_SIZE + 256];')
            lines.append(f'    for (int i = 0; i < BUF_SIZE + 256; i++) {side}_tgt[i] = '
                         f'static_cast<float>((i * 13 + {slot & 0xff}) % 256) * 0.01f + 1.0f;')
            lines.append(f'    int64_t* {side} = new int64_t[BUF_SIZE];')
            lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {side}[i] = '
                         f'reinterpret_cast<int64_t>({side}_tgt + (i % 64) * 4);')
            launch_args.append(f'ArgAt{{0x{slot:x}}}')
            launch_args.append(side)

    if extra_idx_slots is not None:
        lines.append(f'    // idx-scan: {len(extra_idx_slots)} index-buffer slot(s) '
                     'detected in by-value struct param windows')
    if extra_idx_slots:
        # Index arrays (gather/scatter): loaded values feed another
        # pointer's address arithmetic, so they must be ints in [0,N).
        # (i*13+off)%N is bijective per period (13 is coprime to N=32 and
        # to any N not divisible by 13), so scatters cover every target
        # element instead of colliding.
        pslot_fills = (init_hints or {}).get("pslot_fills") or {}
        for slot in sorted(extra_idx_slots):
            side = f'idxslot_{slot:x}'
            if pslot_fills.get(f'0x{slot:x}') == 'zero':
                # Lock/semaphore workspace hint (see ptrptr block): zeroed
                # int buffer; as indices, all-zero is in-range ([0,N)).
                lines.append(f'    int32_t* {side} = new int32_t[BUF_SIZE]();'
                             '  // driver_hints pslot_fills: zero')
                launch_args.append(f'ArgAt{{0x{slot:x}}}')
                launch_args.append(side)
                continue
            # "idx_mod" hint: per-slot index range override. Gather indices
            # default to [0,N) but some kernels bound them by a coupled dim
            # (ELL column blocks: idx < n/blocksize — 43_ell walks
            # idx*blocksize*stride 142MB past B with the default range).
            imod = ((init_hints or {}).get("idx_mod") or {}).get(f'0x{slot:x}')
            mod_expr = str(int(imod)) if imod else '(N > 0 ? N : 1)'
            lines.append(f'    int32_t* {side} = new int32_t[BUF_SIZE];')
            lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {side}[i] = '
                         f'(i * 13 + {slot & 0xff}) % {mod_expr};')
            launch_args.append(f'ArgAt{{0x{slot:x}}}')
            launch_args.append(side)

    if extra_count_slots is not None:
        lines.append(f'    // cnt-scan: {len(extra_count_slots)} grid-barrier '
                     'count slot(s) detected in by-value struct param windows')
    if extra_count_slots:
        # Grid-barrier expected-CTA-count fields: must equal the launched
        # CTA total or the retirement sign-flip never happens and every
        # thread polls the workspace counter forever (see
        # detect_param_ptr_slots fifth class).
        for slot in sorted(extra_count_slots):
            launch_args.append(f'ArgAt{{0x{slot:x}}}')
            launch_args.append('gridDim * gridDimY' if two_d else 'gridDim')

    # buf_init emission (collected above; see init_hints docstring). Each
    # entry allocates a BUF_SIZE side buffer, fills it ("pattern" = the
    # pslot data fill, storage16-aware with an optional "elem" override;
    # "zero" = calloc semantics for workspaces/status words), then pins
    # listed 32-bit WORDS inside it — the cudnn singleread grid-barrier
    # counter must be pre-biased to 0x80000000 - launched threads or the
    # arrive/release sign-flip is unreachable on the non-error path.
    # "check_words"/"check_zero_upto" emit post-launch census asserts: the
    # protocol self-restores its workspace (slots+flags zeroed, counter
    # re-armed) and errStatus stays 0, so the row finally FAILs on any
    # protocol breakage instead of printing TEST PASSED unconditionally.
    binit_checks: List[str] = []
    if buf_init:
        lines.append(f'    // driver_hints buf_init: {len(buf_init)} '
                     'protocol-pinned side buffer(s)')
        for off_s in sorted(buf_init, key=lambda s: int(s, 16)):
            off = int(off_s, 16)
            spec = buf_init[off_s] or {}
            side = f'binit_{off:x}'
            fill = spec.get("fill", "pattern")
            elem = spec.get("elem")
            if fill == "zero":
                elem = elem or 'uint32_t'
                lines.append(f'    {elem}* {side} = new {elem}[BUF_SIZE]();')
            elif fill == "pattern":
                if elem is None and storage16:
                    base, mod = storage16
                    elem = 'uint16_t'
                    lines.append(f'    uint16_t* {side} = '
                                 'new uint16_t[BUF_SIZE];')
                    lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) '
                                 f'{side}[i] = static_cast<uint16_t>({base} | '
                                 f'((i * 13 + {off & 0xff}) % {mod}));')
                elif elem in (None, 'float', 'double'):
                    elem = elem or 'float'
                    sfx = 'f' if elem == 'float' else ''
                    lines.append(f'    {elem}* {side} = new {elem}[BUF_SIZE];')
                    lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) '
                                 f'{side}[i] = static_cast<{elem}>('
                                 f'(i * 13 + {off & 0xff}) % 256) * '
                                 f'0.01{sfx} + 1.0{sfx};')
                else:
                    raise ValueError(f'buf_init 0x{off:x}: unsupported elem '
                                     f'{elem!r} for pattern fill')
            else:
                raise ValueError(f'buf_init 0x{off:x}: unknown fill {fill!r}')
            for woff_s in sorted(spec.get("words") or {},
                                 key=lambda s: int(s, 16)):
                woff = int(woff_s, 16)
                expr = spec["words"][woff_s]
                lines.append(f'    reinterpret_cast<uint32_t*>({side})'
                             f'[{woff // 4}] = static_cast<uint32_t>({expr});'
                             '  // driver_hints buf_init word pin')
            launch_args.append(f'ArgAt{{0x{off:x}}}')
            launch_args.append(side)
            zupto = spec.get("check_zero_upto")
            if zupto is not None:
                nb = int(zupto, 16) // 4
                binit_checks.append(
                    f'    for (int w = 0; w < {nb}; w++) '
                    f'if (reinterpret_cast<uint32_t*>({side})[w] != 0u) {{ '
                    f'std::cout << "FAIL: buf_init census {side} word " << w '
                    f'<< " = 0x" << std::hex << '
                    f'reinterpret_cast<uint32_t*>({side})[w] << std::dec '
                    f'<< std::endl; binit_bad = true; break; }}')
            for woff_s in sorted(spec.get("check_words") or {},
                                 key=lambda s: int(s, 16)):
                woff = int(woff_s, 16)
                expr = spec["check_words"][woff_s]
                binit_checks.append(
                    f'    if (reinterpret_cast<uint32_t*>({side})'
                    f'[{woff // 4}] != static_cast<uint32_t>({expr})) {{ '
                    f'std::cout << "FAIL: buf_init census {side}+0x{woff:x}'
                    f' = 0x" << std::hex << reinterpret_cast<uint32_t*>'
                    f'({side})[{woff // 4}] << " want " << '
                    f'static_cast<uint32_t>({expr}) << std::dec << std::endl;'
                    f' binit_bad = true; }}')

    # word_fills: absolute-offset 4-byte scalar pins inside struct windows.
    # Appended LAST so they override the struct filler, the i64-slot writes
    # and any slot-scan fill at the same offset (ArgAt writes apply in
    # order). See the init_hints docstring for the motivating CUTLASS
    # sparse problem_size constraint (k % 64 == 0).
    # Value forms (singleread protocol pins need more than plain u32):
    #   <int>            u32, written verbatim (the original form);
    #   {"f32": <num>}   FLOAT field — the f32 bit pattern is computed here
    #                    and written as u32 (cudnn bn singleread invCount/
    #                    fracUpscale/threshold live in by-value args structs
    #                    the wordwise N filler turns into denormals);
    #   {"expr": "..."}  launch-coupled scalar — emitted verbatim so the
    #                    driver computes it from blockDim/gridDim (e.g.
    #                    perThreadIters = ceil(count/blockDim) must track
    #                    the per-row template blockDim 32/128/512).
    word_fills = (init_hints or {}).get("word_fills") or {}
    if word_fills:
        lines.append(f'    // driver_hints word_fills: {len(word_fills)} '
                     'constraint-pinned u32 field(s)')
        for off_s in sorted(word_fills, key=lambda s: int(s, 16)):
            off = int(off_s, 16)
            val = word_fills[off_s]
            launch_args.append(f'ArgAt{{0x{off:x}}}')
            if isinstance(val, dict) and "f32" in val:
                import struct as _struct
                bits = _struct.unpack('<I',
                                      _struct.pack('<f', float(val["f32"])))[0]
                launch_args.append(f'0x{bits:08x}u /* f32 {val["f32"]!r} */')
            elif isinstance(val, dict) and "expr" in val:
                launch_args.append(f'static_cast<uint32_t>({val["expr"]})')
            else:
                launch_args.append(str(int(val)))

    lines.append('')

    # BUG-039: Inject c[4] device variable population before launchKernel
    if c4_offsets:
        lines.append(_generate_c4_population(c4_offsets))
        lines.append('')

    # Snapshot every pointer before launch. Reference functions may expose
    # read/write outputs that the mangled signature cannot identify, and
    # passing live buffers after launch would feed the reference mutated
    # inputs rather than the kernel's original state.
    if ref_match is not None:
        lines.append('    // Preserve the identical pre-launch state for the CPU reference')
        for kvar, rvar in ref_match.shadow_pairs:
            elem = buffer_elems.get(kvar, ref_match.elem_type or 'float')
            lines.append(f'    {elem}* {rvar} = new {elem}[BUF_SIZE];')
            lines.append(f'    std::memcpy({rvar}, {kvar}, sizeof({elem}) * BUF_SIZE);')
        lines.append('')

    if two_d:
        launch_head = (f'    launchKernel2D({sig.name}, '
                       'Dim3{gridDim, gridDimY}, Dim3{blockDim},')
        launch_bare = (f'    launchKernel2D({sig.name}, '
                       'Dim3{gridDim, gridDimY}, Dim3{blockDim});')
    else:
        launch_head = f'    launchKernel({sig.name}, gridDim, blockDim,'
        launch_bare = f'    launchKernel({sig.name}, gridDim, blockDim);'
    if launch_args:
        args_str = ',\n                          '.join(launch_args)
        lines.append(launch_head)
        lines.append(f'                          {args_str});')
    else:
        lines.append(launch_bare)
    lines.append('')

    if binit_checks:
        lines.append('    // driver_hints buf_init census: the protocol '
                     'must self-restore its workspace')
        lines.append('    bool binit_bad = false;')
        lines.extend(binit_checks)
        lines.append('    if (binit_bad) return 1;')
        lines.append('')

    # Validation: prefer a strict reference compare when reference.h gave us
    # a usable match, otherwise fall back to the smoke check (output != 0
    # AND no NaN/Inf). The strict path catches lifter/driver bugs that the
    # smoke check silently masks (wrong answer with a non-zero finite value).
    strict_emitted = False
    if ref_match is not None and ref_match.output_pairs:
        # Call reference.h's CPU function using the pre-launch shadows.
        lines.append('    // Strict validation: compare lifted output to reference.h CPU implementation')
        bufvar_to_elem = buffer_elems
        # Emit reference function call (with optional template instantiation).
        if ref_match.template_args:
            tpl = '<' + ', '.join(ref_match.template_args) + '>'
        else:
            tpl = ''
        ref_args_str = ', '.join(ref_match.arg_exprs)
        lines.append(f'    {ref_match.func.name}{tpl}({ref_args_str});')
        lines.append('')
        # Compare loop. We only check the first N*blockDim elements
        # because the kernel may not write past its grid bound, and
        # the ref didn't either.
        lines.append('    // Compare element-wise (tolerance for floats, exact for ints).')
        lines.append('    // Cap at N because the reference function only iterates N times,')
        lines.append('    // and most CUDA kernels guard with `if (tid < N)`. blockDim*gridDim')
        lines.append('    // can exceed N, in which case the tail is uninitialised on both sides.')
        lines.append('    int n_diff = 0;')
        lines.append('    int first_diff = -1;')
        lines.append('    int compare_n = N < blockDim * gridDim ? N : blockDim * gridDim;')
        lines.append('    if (compare_n > BUF_SIZE) compare_n = BUF_SIZE;')
        for kvar, rvar in ref_match.output_pairs:
            elem = bufvar_to_elem.get(kvar, ref_match.elem_type or 'float')
            if elem in ('float', 'double'):
                lines.append(f'    for (int i = 0; i < compare_n; i++) {{')
                lines.append(f'        {elem} a = {kvar}[i];')
                lines.append(f'        {elem} b = {rvar}[i];')
                lines.append(f'        {elem} d = (a > b) ? (a - b) : (b - a);')
                lines.append(f'        {elem} scale = std::fabs(b) > {elem}(1) ? std::fabs(b) : {elem}(1);')
                lines.append(f'        if (std::isnan(a) || std::isinf(a) || d / scale > {elem}(1e-3)) {{')
                lines.append(f'            if (first_diff < 0) first_diff = i;')
                lines.append(f'            n_diff++;')
                lines.append(f'        }}')
                lines.append(f'    }}')
            else:
                lines.append(f'    for (int i = 0; i < compare_n; i++) {{')
                lines.append(f'        if ({kvar}[i] != {rvar}[i]) {{')
                lines.append(f'            if (first_diff < 0) first_diff = i;')
                lines.append(f'            n_diff++;')
                lines.append(f'        }}')
                lines.append(f'    }}')
        lines.append('    if (n_diff > 0) {')
        # Print the first failing index/values for the FIRST output pair (best diagnostics).
        first_kvar, first_rvar = ref_match.output_pairs[0]
        lines.append(f'        std::cout << "FAIL: " << n_diff << "/" << compare_n')
        lines.append(f'                  << " elements differ; first at " << first_diff')
        lines.append(f'                  << " got=" << static_cast<double>({first_kvar}[first_diff])')
        lines.append(f'                  << " ref=" << static_cast<double>({first_rvar}[first_diff])')
        lines.append(f'                  << std::endl;')
        lines.append('        return 1;')
        lines.append('    }')
        lines.append('')
        strict_emitted = True

    if output_bufs and not strict_emitted:
        lines.append('    // Validation: check output is not all zeros (smoke test)')
        lines.append('    bool ok = false;')
        for var, elem in output_bufs:
            zero_check = f'{var}[i] != 0' if elem not in ('float', 'double') else f'{var}[i] != 0.0f'
            lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {{')
            lines.append(f'        if ({zero_check}) {{ ok = true; break; }}')
            lines.append(f'    }}')
        lines.append('    if (!ok) {')
        lines.append('        std::cout << "FAIL: output is all zeros" << std::endl;')
        lines.append('        return 1;')
        lines.append('    }')
        lines.append('')
        if (init_hints or {}).get("allow_nan"):
            # Per-test waiver (driver_hints.json): some kernels GENUINELY
            # produce NaN/Inf — verified by compiling the kernel's own math
            # as host C++ and feeding it the real app's inputs (lci-cuda
            # NaNs at l=24 either way). Rejecting NaN there flags correct
            # lifting as wrong_result.
            lines.append('    // NaN/Inf check waived by driver_hints (genuine app behavior)')
            lines.append('')
            emit_nan_check = False
        else:
            emit_nan_check = True
        if emit_nan_check:
            lines.append('    // Check for NaN/Inf')
            lines.append('    bool has_nan = false;')
            for var, elem in output_bufs:
                if elem in ('float', 'double'):
                    lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {{')
                    lines.append(f'        if (std::isnan({var}[i]) || std::isinf({var}[i])) {{')
                    lines.append(f'            std::cout << "FAIL: NaN/Inf at index " << i << std::endl;')
                    lines.append(f'            has_nan = true; break;')
                    lines.append(f'        }}')
                    lines.append(f'    }}')
            lines.append('    if (has_nan) return 1;')
            lines.append('')

    lines.append('    std::cout << "TEST PASSED" << std::endl;')
    lines.append('')

    # Cleanup
    for i in range(buf_idx):
        lines.append(f'    delete[] buf{i};')
    if ref_match is not None:
        for _, rvar in ref_match.shadow_pairs:
            lines.append(f'    delete[] {rvar};')
    # BUG-039: cleanup c[4] heap buffers
    if c4_offsets:
        for i in range(len(c4_offsets)):
            lines.append(f'    delete[] c4_buf{i};')

    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def generate_and_save(sig: KernelSignature, output_dir: str, sm: int = 90,
                      sm_mode: str = "normal", c4_offsets: Set[int] = None):
    """Generate host_cpu.cpp and save to output_dir."""
    content = generate_host(sig, sm, sm_mode=sm_mode, c4_offsets=c4_offsets)
    out_path = Path(output_dir) / "host_cpu.cpp"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    return out_path


def _elem_for_param(p) -> str:
    """Pick the host-side allocation element type for a pointer param,
    mirroring the choice generate_host makes. Centralised so both x86
    and CUDA emitters agree."""
    ct = p.ctype.replace('const ', '')
    if '__half' in ct or 'bfloat' in ct or '__nv_bfloat16' in ct:
        # 16-bit float buffers: allocate as raw 16-bit ints (host has no
        # native half); init values are small bit patterns, harmless.
        return 'uint16_t'
    if 'double' in ct:
        return 'double'
    if 'long long' in ct or 'long' in ct:
        return 'int64_t'
    if 'short' in ct:
        return 'int16_t'
    if 'char' in ct:
        return 'int8_t'
    if 'int' in ct:
        return 'int'
    if 'bool' in ct:
        return 'int8_t'
    return 'float'


def generate_host_cuda(sig: KernelSignature, sm: int = 90, sm_mode: str = "normal",
                       ref_match=None) -> str:
    """Generate a host_cuda.cu driver that runs the lifted kernel via the
    CUDA driver API and (optionally) compares against reference.h.

    The driver:
      * takes a cubin path as argv[1],
      * allocates host buffers with the same prime/offset init as generate_host,
      * copies inputs to device,
      * launches the kernel via cuLaunchKernel,
      * copies outputs back,
      * runs the matched reference function on host,
      * compares element-wise with a 1e-3 tolerance for floats.

    Buffers we'd run out of GPU memory for at BUF_SIZE=16M get scaled down
    to N when sm_mode != 'normal'.
    """
    # Bail early on by-value structs — packing those for cuLaunchKernel is
    # currently unsupported in this generator.
    if any(getattr(p, 'is_byval_struct', False) for p in sig.params):
        return ""

    lines: List[str] = []
    lines.append('#include <cstdint>')
    lines.append('#include <iostream>')
    lines.append('#include <cmath>')
    lines.append('#include <cstring>')
    lines.append('#include <vector>')
    lines.append('#include <algorithm>')
    lines.append('#include <cfloat>')
    lines.append('#include <climits>')
    lines.append('#include <cstdlib>')
    lines.append('#include <unistd.h>')
    lines.append('#include "cuda_driver_helpers.h"')
    lines.append('')

    # Inline reference.h with type stubs (same approach as generate_host).
    if ref_match is not None and (ref_match.header_source or ref_match.func.source):
        body = ref_match.header_source or ref_match.func.source
        stubs = []
        if re.search(r"\b__half\b|\bhalf\b", body):
            stubs.append('struct __slf_half_stub { float v; '
                         '__slf_half_stub() : v(0) {} '
                         '__slf_half_stub(float f) : v(f) {} '
                         'operator float() const { return v; } };')
            stubs.append('typedef __slf_half_stub __half;')
            stubs.append('typedef __slf_half_stub half;')
        if re.search(r"\b__nv_bfloat16\b|\bbfloat16\b", body):
            stubs.append('typedef __slf_half_stub __nv_bfloat16;')
            stubs.append('typedef __slf_half_stub bfloat16;')
        for s in stubs:
            lines.append(s)
        if stubs:
            lines.append('')
        lines.append(body)
        lines.append('')

    # The PTX entry name. SLifter preserves the lifted symbol (mangled CUDA
    # name) end-to-end; nvcc's PTX writer emits it verbatim as the .entry name.
    lines.append(f'static constexpr const char* kKernelName = "{sig.name}";')
    lines.append('')
    lines.append('int main(int argc, char** argv) {')
    lines.append('    if (argc < 2) {')
    lines.append('        std::cerr << "usage: " << argv[0] << " <cubin>\\n";')
    lines.append('        return 1;')
    lines.append('    }')

    if sm_mode == "small":
        lines.append('    constexpr int N = 1;')
        lines.append('    constexpr int BUF_SIZE = 1024;')
        lines.append('    constexpr int blockDim = 1;')
    else:
        lines.append('    constexpr int N = 32;')
        # GPU memory is plentiful but allocating 16M elements per buffer for
        # 8-arg kernels still exceeds H100 free RAM; cap smaller than x86.
        lines.append('    constexpr int BUF_SIZE = 1048576;')
        lines.append('    constexpr int blockDim = 256;')
    lines.append('    constexpr int gridDim = (N + blockDim - 1) / blockDim;')
    lines.append('')
    # Init CUDA + load module BEFORE any cuMemAlloc.
    lines.append('    CuDriverSession session;')
    lines.append('    CUfunction fn = session.loadKernel(argv[1], kKernelName);')
    lines.append('')

    # Allocate host buffers + scalars and copy buffers to device.
    buf_idx = 0
    output_bufs = []  # (var, elem)
    buffer_elems = {}
    launch_bind = []  # ordered (kind, expr) for params[]: kind in {ptr, scalar}
    cleanup_d = []

    for p in sig.params:
        if p.is_pointer:
            var = f"buf{buf_idx}"
            d_var = f"d_{var}"
            elem = _elem_for_param(p)
            prime = _BUF_PRIMES[buf_idx % len(_BUF_PRIMES)]
            offset = _BUF_OFFSETS[buf_idx % len(_BUF_OFFSETS)]
            buf_idx += 1
            lines.append(f'    {elem}* {var} = new {elem}[BUF_SIZE];')
            if elem == 'float':
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {var}[i] = static_cast<float>((i * {prime} + {offset}) % 256) * 0.01f + 1.0f;')
            elif elem == 'double':
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {var}[i] = static_cast<double>((i * {prime} + {offset}) % 256) * 0.01 + 1.0;')
            else:
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {var}[i] = static_cast<{elem}>((i * {prime} + {offset}) % (N > 0 ? N : 1));')
            lines.append(f'    CUdeviceptr {d_var}{{}};')
            lines.append(f'    CUCHK(cuMemAlloc(&{d_var}, sizeof({elem}) * BUF_SIZE));')
            lines.append(f'    CUCHK(cuMemcpyHtoD({d_var}, {var}, sizeof({elem}) * BUF_SIZE));')
            launch_bind.append(('ptr', d_var))
            cleanup_d.append(d_var)
            buffer_elems[var] = elem
            if p.is_output:
                output_bufs.append((var, elem, d_var))
        else:
            # Scalar: emit a typed local so we can take its address.
            sname = f"s{len([b for b in launch_bind if b[0] == 'scalar'])}"
            if p.ctype == 'float':
                lines.append(f'    float {sname} = 2.0f;')
            elif p.ctype == 'double':
                lines.append(f'    double {sname} = 2.0;')
            elif p.ctype == 'bool':
                lines.append(f'    bool {sname} = true;')
            elif p.ctype == 'int64_t':
                lines.append(f'    int64_t {sname} = static_cast<int64_t>(N);')
            else:
                lines.append(f'    int {sname} = N;')
            launch_bind.append(('scalar', sname))

    lines.append('')
    if ref_match is not None:
        lines.append('    // Preserve the identical pre-launch state for the CPU reference')
        for kvar, rvar in ref_match.shadow_pairs:
            elem = buffer_elems.get(kvar, ref_match.elem_type or 'float')
            lines.append(f'    {elem}* {rvar} = new {elem}[BUF_SIZE];')
            lines.append(f'    std::memcpy({rvar}, {kvar}, sizeof({elem}) * BUF_SIZE);')
        lines.append('')

    # Pack params[] for cuLaunchKernel — entries are pointers to each arg.
    if launch_bind:
        lines.append('    void* params[] = {')
        for kind, expr in launch_bind:
            lines.append(f'        (void*)&{expr},')
        lines.append('    };')
    else:
        lines.append('    void** params = nullptr;')
    lines.append('')
    # Launch on a non-default stream so we can poll-wait with a deadline
    # instead of doing an unbounded cuCtxSynchronize. If a lifted kernel
    # spins forever on the device, an unbounded sync leaves the host
    # process stuck in a kernel-uninterruptible state — even SIGKILL
    # can\'t free its UVM context, which then wedges every subsequent
    # test launched on the same GPU. Polling lets us bail with exit(2)
    # before that happens. Doesn\'t unstick the device, but keeps the
    # host PID killable so a parent runner\'s killpg actually works.
    lines.append('    CUstream stream{};')
    lines.append('    CUCHK(cuStreamCreate(&stream, 0));')
    lines.append('    CUCHK(cuLaunchKernel(')
    lines.append('        fn,')
    lines.append('        gridDim, 1, 1,')
    lines.append('        blockDim, 1, 1,')
    lines.append('        0,')
    lines.append('        stream,')
    lines.append('        params,')
    lines.append('        nullptr));')
    lines.append('    {')
    lines.append('        // Poll the stream with a 20s wall-clock deadline.')
    lines.append('        const int deadline_ms = 20000;')
    lines.append('        const int poll_us = 1000;')
    lines.append('        int waited_us = 0;')
    lines.append('        CUresult st;')
    lines.append('        while ((st = cuStreamQuery(stream)) == CUDA_ERROR_NOT_READY) {')
    lines.append('            usleep(poll_us);')
    lines.append('            waited_us += poll_us;')
    lines.append('            if (waited_us > deadline_ms * 1000) {')
    lines.append('                std::cerr << "FAIL: kernel hung past " << deadline_ms')
    lines.append('                          << "ms; aborting" << std::endl;')
    lines.append('                std::exit(2);')
    lines.append('            }')
    lines.append('        }')
    lines.append('        if (st != CUDA_SUCCESS) {')
    lines.append('            const char* en = nullptr;')
    lines.append('            cuGetErrorName(st, &en);')
    lines.append('            std::cerr << "FAIL: cuStreamQuery returned "')
    lines.append('                      << (en ? en : "<unknown>") << std::endl;')
    lines.append('            std::exit(3);')
    lines.append('        }')
    lines.append('    }')
    lines.append('')

    # Copy outputs back to host.
    for var, elem, d_var in output_bufs:
        lines.append(f'    CUCHK(cuMemcpyDtoH({var}, {d_var}, sizeof({elem}) * BUF_SIZE));')
    lines.append('')

    # Validation: strict if matched, else smoke (output != 0, no NaN/Inf).
    strict_emitted = False
    if ref_match is not None and ref_match.output_pairs:
        bufvar_to_elem = buffer_elems
        lines.append('    // Strict validation: compare against reference.h CPU impl')
        if ref_match.template_args:
            tpl = '<' + ', '.join(ref_match.template_args) + '>'
        else:
            tpl = ''
        ref_args_str = ', '.join(ref_match.arg_exprs)
        lines.append(f'    {ref_match.func.name}{tpl}({ref_args_str});')
        lines.append('')
        lines.append('    int n_diff = 0;')
        lines.append('    int first_diff = -1;')
        lines.append('    int compare_n = N < blockDim * gridDim ? N : blockDim * gridDim;')
        lines.append('    if (compare_n > BUF_SIZE) compare_n = BUF_SIZE;')
        for kvar, rvar in ref_match.output_pairs:
            elem = bufvar_to_elem.get(kvar, ref_match.elem_type or 'float')
            if elem in ('float', 'double'):
                lines.append(f'    for (int i = 0; i < compare_n; i++) {{')
                lines.append(f'        {elem} a = {kvar}[i];')
                lines.append(f'        {elem} b = {rvar}[i];')
                lines.append(f'        {elem} d = (a > b) ? (a - b) : (b - a);')
                lines.append(f'        {elem} scale = std::fabs(b) > {elem}(1) ? std::fabs(b) : {elem}(1);')
                lines.append(f'        if (std::isnan(a) || std::isinf(a) || d / scale > {elem}(1e-3)) {{')
                lines.append(f'            if (first_diff < 0) first_diff = i;')
                lines.append(f'            n_diff++;')
                lines.append(f'        }}')
                lines.append(f'    }}')
            else:
                lines.append(f'    for (int i = 0; i < compare_n; i++) {{')
                lines.append(f'        if ({kvar}[i] != {rvar}[i]) {{')
                lines.append(f'            if (first_diff < 0) first_diff = i;')
                lines.append(f'            n_diff++;')
                lines.append(f'        }}')
                lines.append(f'    }}')
        lines.append('    if (n_diff > 0) {')
        first_kvar, first_rvar = ref_match.output_pairs[0]
        lines.append(f'        std::cout << "FAIL: " << n_diff << "/" << compare_n')
        lines.append(f'                  << " elements differ; first at " << first_diff')
        lines.append(f'                  << " got=" << static_cast<double>({first_kvar}[first_diff])')
        lines.append(f'                  << " ref=" << static_cast<double>({first_rvar}[first_diff])')
        lines.append(f'                  << std::endl;')
        lines.append('        return 1;')
        lines.append('    }')
        strict_emitted = True

    if output_bufs and not strict_emitted:
        lines.append('    // Smoke validation: output is not all zeros and has no NaN/Inf')
        lines.append('    bool ok = false;')
        for var, elem, _ in output_bufs:
            zero = f'{var}[i] != 0' if elem not in ('float', 'double') else f'{var}[i] != 0.0f'
            lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {{ if ({zero}) {{ ok = true; break; }} }}')
        lines.append('    if (!ok) { std::cout << "FAIL: output is all zeros" << std::endl; return 1; }')
        lines.append('    bool has_nan = false;')
        for var, elem, _ in output_bufs:
            if elem in ('float', 'double'):
                lines.append(f'    for (int i = 0; i < BUF_SIZE; i++) {{ if (std::isnan({var}[i]) || std::isinf({var}[i])) {{ has_nan = true; break; }} }}')
        lines.append('    if (has_nan) { std::cout << "FAIL: NaN/Inf" << std::endl; return 1; }')

    lines.append('')
    lines.append('    std::cout << "TEST PASSED" << std::endl;')

    # Cleanup.
    for d_var in cleanup_d:
        lines.append(f'    cuMemFree({d_var});')
    for i in range(buf_idx):
        lines.append(f'    delete[] buf{i};')
    if ref_match is not None:
        for _, rvar in ref_match.shadow_pairs:
            lines.append(f'    delete[] {rvar};')

    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines) + '\n'
