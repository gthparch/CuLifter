"""Repair stale phi entries in emitted LLVM IR text.

The lifter's emitted CFG occasionally diverges from what the SSA pass saw —
causes include placeholder `ret void` for unsupported control-flow ops,
direct CALL.REL.NOINC lifted as a branch to a different target, R2P-bypass
predicate writes, etc. The result: phi nodes whose incoming-block list
doesn't match the actual predecessor set of their parent BB. LLVM rejects
this with "PHINode should have one entry for each predecessor of its
parent basic block!".

`repair_stale_phi_entries(text)` walks the text IR, builds the true
predecessor relation from terminator instructions, then rewrites each
phi node to (a) drop entries from non-predecessor BBs and (b) add `undef`
entries for predecessors that have no existing entry. The latter can
produce wrong runtime results if the missing predecessor is genuinely
live, but it satisfies LLVM's structural verification and lets the rest
of the kernel run.

A proper fix lives in the SSA pass (re-run it after every CFG-modifying
lift step), but that's invasive — this post-processor is the pragmatic
workaround.
"""
import re


# BB labels can contain dots — the lifter emits `EntryBB_0000.1`,
# `BB_03f0.2`, etc., for re-emitted blocks across multiple kernels in
# the same module. Without `.` in the character class, my pass1 fails
# to detect those labels, ascribes their content to whichever BB came
# before, and ends up with absurd predecessor sets — e.g. an
# unreachable `ret void` BB looking like the predecessor of every
# branch in the kernel. Apply this fix to every BB-name regex.
_BB_NAME_CHARS = r"[A-Za-z0-9_.]+"

_BB_LABEL_RE = re.compile(rf"^({_BB_NAME_CHARS}):\s*$")
_FN_BEGIN_RE = re.compile(r"^define\b")
_FN_END_RE = re.compile(r"^}\s*$")
_PHI_PREFIX_RE = re.compile(
    rf'(\s*%"?{_BB_NAME_CHARS}"?\s*=\s*phi\s+)(\S+)(\s*)(.*)'
)
_PHI_ENTRY_RE = re.compile(
    rf'\[\s*([^,\]]+)\s*,\s*%"?({_BB_NAME_CHARS})"?\s*\]'
)


def repair_stale_phi_entries(text: str) -> str:
    lines = text.split("\n")

    # The CFG must be analysed per-function — two functions can have BBs
    # with the same label, and cross-function predecessor confusion was
    # the cause of regressions when this pass was first introduced.
    # First split lines into per-function ranges.
    func_ranges = []     # list of (start_idx, end_idx_inclusive)
    in_func = False
    fn_start = None
    for i, ln in enumerate(lines):
        if not in_func and _FN_BEGIN_RE.match(ln):
            in_func = True
            fn_start = i
        elif in_func and _FN_END_RE.match(ln):
            func_ranges.append((fn_start, i))
            in_func = False
            fn_start = None

    out = list(lines)
    for fn_start, fn_end in func_ranges:
        fn_lines = lines[fn_start:fn_end + 1]
        repaired = _repair_function(fn_lines)
        out[fn_start:fn_end + 1] = repaired
    return "\n".join(out)


def _repair_function(lines):
    # Pass 1: BB -> set of successor BB names (from terminator instructions),
    # within this function only.
    bb_succs = {}
    cur_bb = None
    for ln in lines:
        m = _BB_LABEL_RE.match(ln)
        if m:
            cur_bb = m.group(1)
            bb_succs[cur_bb] = set()
            continue
        if cur_bb is None:
            continue
        stripped = ln.strip()
        if stripped.startswith("br "):
            for tgt in re.findall(rf'%"?({_BB_NAME_CHARS})"?', stripped):
                bb_succs[cur_bb].add(tgt)
        elif stripped.startswith("switch "):
            for tgt in re.findall(rf'label\s+%"?({_BB_NAME_CHARS})"?', stripped):
                bb_succs[cur_bb].add(tgt)

    bb_names = set(bb_succs.keys())
    for bb in list(bb_succs.keys()):
        bb_succs[bb] &= bb_names

    # Pass 2: invert to predecessor map.
    bb_preds = {bb: set() for bb in bb_names}
    for src, succs in bb_succs.items():
        for s in succs:
            if s in bb_preds:
                bb_preds[s].add(src)

    # Pass 3: rewrite phi nodes.
    out = []
    cur_bb = None
    for ln in lines:
        m = _BB_LABEL_RE.match(ln)
        if m:
            cur_bb = m.group(1)
            out.append(ln)
            continue
        if cur_bb is None or "phi" not in ln:
            out.append(ln)
            continue
        prefix_m = _PHI_PREFIX_RE.match(ln)
        if not prefix_m:
            out.append(ln)
            continue
        prefix, phi_ty, gap, rest = prefix_m.groups()
        true_preds = bb_preds.get(cur_bb, set())
        entries = _PHI_ENTRY_RE.findall(rest)
        kept = [(v, bb) for (v, bb) in entries if bb in true_preds]
        kept_bbs = {bb for (_v, bb) in kept}
        for missing in sorted(true_preds - kept_bbs):
            kept.append(("undef", missing))
        if not kept:
            out.append(ln)
            continue
        new_rest = ", ".join(f'[{v}, %"{bb}"]' for (v, bb) in kept)
        out.append(prefix + phi_ty + gap + new_rest)
    return out
