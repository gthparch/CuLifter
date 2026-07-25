"""Cross-Function CALL-ABI inlining.

SASS produced for `__device__` helper calls (e.g. fresnel-cuda's
`Fresnel_Sine_Integral(x)`) places each callee in its OWN top-level `.text`
section, parsed by SLifter into a SEPARATE top-level `Function`.  The call is a
`CALL.ABS.NOINC 0x0` whose real target lives in an ELF relocation
(`.rela.text.<caller>`, type `R_CUDA_ABS55_16_34`) naming the callee function
symbol.  The SASS call ABI passes arguments and the return value in FIXED
registers (e.g. R4:R5 for an f64 in/out) — there is no per-call argument
marshalling.

`HandleCall` already inlines call-only sub-CFGs that live WITHIN one parsed
`Function` (intra-function device helpers share that Function's address space).
It does NOT, however, reach across top-level `Function`s, and the per-function
transform pipeline (`_transform_per_function`) further isolates each Function so
the siblings aren't even visible.  The result: the cross-function call lifts as
an argless `void()` CALLFUNC, gets DCE'd, and the entire helper computation
vanishes — fresnel collapses to an identity copy.

This MODULE-LEVEL pass runs BEFORE the per-function pipeline (while every
Function is still a flat, untransformed instruction list).  For each kernel it
collects the acyclic set of callee Functions reachable through the CALL-target
relocations, then merges each callee's instructions into the kernel's flat
instruction list at a DISJOINT, rebased address range (rewriting the callees'
internal branch / call targets accordingly) and rewrites the kernel's
`CALL.ABS.NOINC 0x0` into a direct `CALL.ABS.NOINC <rebased-entry>`.

The merged callees then appear to `BuildCFG`/`HandleCall` exactly like
intra-function device helpers — disconnected CFG components reachable only via a
direct CALL — and the existing intra-function inline path lowers them at every
call site with the live register namespace intact (R4:R5 in/out needs no
marshalling because the callee instructions already reference those registers).

Recursion (self- or mutual-) is bounded by collecting only an ACYCLIC reachable
set per kernel: a callee already on the current DFS chain is left as an
unresolved CALL (the intra-function path's existing cycle handling then bounds
its expansion, as it does for frechet-cuda's recursive distance_norm3 chain).
"""

from transform.transform import SaSSTransform
from sir.operand import Operand


class CrossFuncInline(SaSSTransform):
    # Address gap between rebased callee regions. Must exceed any single
    # callee's instruction span so regions never overlap. fresnel's largest
    # callee is ~5.7 KB of text; 0x100000 (1 MiB) per region is comfortably
    # beyond any realistic device helper.
    REGION_STRIDE = 0x100000

    def apply(self, module):
        elf = getattr(module, "elf", None)
        if elf is None:
            return

        call_map = getattr(elf, "rel_text_call_map", None)
        if not call_map:
            return

        # symbol name -> Function. The same helper can be emitted multiple
        # times (e.g. __cuda_sm20_div_rn_f64_full / __1 / __2); the parser
        # suffixes duplicates but the relocation names the bare symbol, so map
        # the bare name to the FIRST matching Function (the copies are
        # identical).
        func_by_name = {}
        for func in module.functions:
            func_by_name.setdefault(func.name, func)

        # Launch kernels are the inline ROOTS; helper-to-helper edges are then
        # followed transitively from each root. The ELF emits a `.nv.constant0`
        # param descriptor for EVERY top-level function (device helpers
        # included), so presence in arg_map is not enough — only genuine launch
        # entries have a NON-EMPTY parameter list (device helpers pass args in
        # registers, so their arg_map entry is []). That non-empty-params test
        # is the precise launch-kernel discriminator.
        arg_map = getattr(elf, "arg_map", {}) or {}
        kernels = [f for f in module.functions
                   if arg_map.get(f.name) and f.name in call_map]
        if not kernels:
            return

        inlined_any = False
        for kernel in kernels:
            if self._inline_into_kernel(kernel, call_map, func_by_name, elf):
                inlined_any = True

        if not inlined_any:
            return

        # Drop the now-merged helper Functions so they aren't lifted twice
        # (their bodies now live inside each kernel that called them). Keep
        # kernels and any Function that was never reached from a kernel (it may
        # be an independently-launchable entry, or a helper for a cycle we left
        # unresolved — those still need their own definition).
        # A helper emitted multiple times is parser-suffixed (`__1`, `__2`);
        # the relocation names the bare symbol so only the bare-named copy was
        # merged. Drop the orphan duplicates too — they share the merged body
        # and are otherwise lifted as dead, param-less standalone "kernels".
        merged_bare = set(self._merged_helpers)

        def _bare(name):
            i = name.rfind("__")
            if i > 0 and name[i + 2:].isdigit():
                return name[:i]
            return name

        keep = []
        for func in module.functions:
            if arg_map.get(func.name):
                keep.append(func)
                continue
            if func.name in merged_bare or _bare(func.name) in merged_bare:
                continue
            keep.append(func)
        module.functions = keep

    def _inline_into_kernel(self, kernel, call_map, func_by_name, elf=None):
        """Merge the acyclic callee closure into `kernel`. Returns True if any
        callee body was merged."""
        self._merged_helpers = getattr(self, "_merged_helpers", set())

        # Assign each reachable callee a disjoint rebase base via DFS over the
        # CALL-target relocations, skipping back-edges (cycles).
        bases = {}            # func name -> rebase base address
        on_stack = set()      # names on the current DFS chain (cycle guard)
        next_base = [self._region_start(kernel)]

        def visit(func):
            """Assign bases to every acyclic callee reachable from func."""
            on_stack.add(func.name)
            for sym in self._callee_symbols(func, call_map):
                callee = func_by_name.get(sym)
                if callee is None or callee is kernel:
                    continue
                if callee.name in on_stack:
                    # Back edge -> recursion. Leave as an unresolved CALL.
                    continue
                if callee.name not in bases:
                    bases[callee.name] = next_base[0]
                    next_base[0] += self.REGION_STRIDE
                    visit(callee)
            on_stack.discard(func.name)

        visit(kernel)
        if not bases:
            return False

        merged_insts = []
        for name, base in bases.items():
            callee = func_by_name[name]
            merged_insts.extend(
                self._clone_rebased(callee, base, bases, call_map, func_by_name))
            self._merged_helpers.add(name)
            # Merge the callee's ABS32 device-global relocations into the
            # kernel's relmap, rebasing each MOV offset by `base` (mirrors the
            # instruction-id rebase in _clone_rebased). Without this, an inlined
            # coefficient-table base load (`MOV Rlo,0x0`/`MOV Rhi,0x0`) can't be
            # resolved by the lifter — its zero immediate stays 0 and the table
            # null-derefs (fresnel: Chebyshev/trig-reduction coeff OOB read).
            self._merge_abs32_relocs(elf, kernel.name, name, base)

        # Rewrite the kernel's own cross-function CALLs to direct CALLs at the
        # rebased callee entry, then append the merged callee bodies.
        self._rewrite_calls(kernel, base=0, bases=bases, call_map=call_map,
                            func_by_name=func_by_name)
        # Strip the kernel's trailing dead padding (the post-EXIT `BRA <self>`
        # and NOP fill). Otherwise that NOP block ends without a terminator and
        # BuildCFG wires a spurious fall-through edge into the FIRST appended
        # callee block — fusing the callee region into the kernel's main CFG so
        # extract_device_functions can no longer see it as a separate
        # (inlinable) component, and the kernel CALL is left unresolved.
        kernel.instructions = (self._strip_trailing_padding(kernel.instructions)
                               + merged_insts)
        return True

    @staticmethod
    def _merge_abs32_relocs(elf, kernel_name, callee_name, base):
        """Copy callee_name's R_CUDA_ABS32 device-global relocations into
        kernel_name's relmap, rebasing each MOV text-offset by `base` to match
        the inlined instruction ids. No-op if there's nothing to merge."""
        if elf is None:
            return
        relmaps = getattr(elf, "rel_text_abs32", None)
        if not relmaps:
            return
        callee_map = relmaps.get(callee_name)
        if not callee_map:
            return
        kernel_map = relmaps.setdefault(kernel_name, {})
        for off, ent in callee_map.items():
            # off is already 16-byte aligned (lifter masks with ~0xF too).
            kernel_map.setdefault((base + off) & ~0xF, ent)

    @staticmethod
    def _strip_trailing_padding(insts):
        """Drop trailing NOPs and a final self-looping BRA after the region's
        last real terminator (EXIT/RET). These carry no semantics and, left in
        place, fall through into a concatenated region."""
        out = list(insts)
        while out:
            last = out[-1]
            op = last.opcodes[0] if last.opcodes else None
            if op == "NOP":
                out.pop()
                continue
            if op == "BRA" and last.operands:
                tgt = last.operands[-1]
                # Self-loop `BRA <own addr>` (trap padding after EXIT).
                if (getattr(tgt, "is_immediate", False)
                        and tgt.immediate_value is not None):
                    try:
                        if tgt.immediate_value == int(last.id, 16):
                            out.pop()
                            continue
                    except (ValueError, TypeError):
                        pass
            break
        return out

    def _region_start(self, kernel):
        """First rebase base: comfortably above the kernel's own addresses."""
        max_addr = 0
        for inst in kernel.instructions:
            try:
                max_addr = max(max_addr, int(inst.id, 16))
            except (ValueError, TypeError):
                continue
        start = ((max_addr // self.REGION_STRIDE) + 1) * self.REGION_STRIDE
        return start

    def _callee_symbols(self, func, call_map):
        """Distinct callee symbols this func calls (via CALL-target relocs)."""
        offsets = call_map.get(func.name) or {}
        syms = []
        seen = set()
        for inst in func.instructions:
            if not inst.is_call() or "ABS" not in inst.opcodes:
                continue
            off = self._call_offset(inst)
            if off is None:
                continue
            sym = offsets.get(off)
            if sym and sym not in seen:
                seen.add(sym)
                syms.append(sym)
        return syms

    @staticmethod
    def _call_offset(inst):
        try:
            return int(inst.id, 16)
        except (ValueError, TypeError):
            return None

    def _clone_rebased(self, callee, base, bases, call_map, func_by_name):
        """Clone callee instructions, rebasing their ids and internal
        branch/call targets by `base`, and rewriting cross-function CALLs to
        direct CALLs at the called helper's own rebased entry."""
        cloned = []
        # Drop the callee's trailing dead padding before rebasing so it can't
        # fall through into the next concatenated region (same hazard as the
        # kernel tail handled by _strip_trailing_padding).
        for inst in self._strip_trailing_padding(callee.instructions):
            ci = inst.clone()
            try:
                ci.id = format(base + int(inst.id, 16), "04x")
            except (ValueError, TypeError):
                pass
            # Rebase internal branch targets (BRA/PBRA) by `base`.
            if ci.is_branch() and not ci.is_indirect_branch():
                for op in self._branch_target_ops(ci):
                    self._rebase_target(op, base)
            cloned.append(ci)
        # Rewrite this callee's cross-function CALLs (to OTHER helpers).
        self._rewrite_calls(_FlatFunc(callee.name, cloned), base, bases,
                            call_map, func_by_name)
        return cloned

    def _rewrite_calls(self, func, base, bases, call_map, func_by_name):
        """Turn each cross-function `CALL.ABS.NOINC 0x0` in func into a direct
        CALL at the target helper's rebased entry. `base` is func's own rebase
        offset (0 for a kernel); the reloc offset is the call's ORIGINAL
        (un-rebased) text offset."""
        offsets = call_map.get(func.name) or {}
        for inst in func.instructions:
            if not inst.is_call() or "ABS" not in inst.opcodes:
                continue
            try:
                orig_off = int(inst.id, 16) - base
            except (ValueError, TypeError):
                continue
            sym = offsets.get(orig_off)
            if not sym:
                continue
            callee = func_by_name.get(sym)
            if callee is None or sym not in bases:
                # Unresolved (cycle back-edge or missing) — leave as-is; the
                # intra-function path / lifter handles the residual CALL.
                continue
            target = bases[sym]
            self._set_call_target(inst, target)

    @staticmethod
    def _branch_target_ops(inst):
        if inst.opcodes and inst.opcodes[0] == "PBRA":
            ops = []
            if len(inst.operands) > 1:
                ops.append(inst.operands[1])
            if len(inst.operands) > 2:
                ops.append(inst.operands[2])
            return ops
        return [inst.operands[-1]] if inst.operands else []

    @staticmethod
    def _rebase_target(op, base):
        if op is None or not getattr(op, "is_immediate", False):
            return
        if op.immediate_value is None or op.is_reg:
            return
        new_val = op.immediate_value + base
        op.immediate_value = new_val
        # Keep the `0x` prefix: Operand.is_float_immediate treats a prefix-less
        # name as a float, which makes __str__ emit a decimal that BuildCFG then
        # mis-parses as hex (round-trip corruption).
        op.name = f"0x{new_val:x}"

    @staticmethod
    def _set_call_target(inst, target_addr):
        new_op = Operand.from_immediate(f"0x{target_addr:x}", target_addr)
        inst.operands[0] = new_op
        # CALL.ABS.NOINC with an immediate target is treated as a direct call
        # by BuildCFG (adds the target as a block leader) and HandleCall.
        inst.inst_content = (
            f"{'.'.join(inst.opcodes)} 0x{target_addr:x} ;")


class _FlatFunc:
    """Minimal func-like wrapper so _rewrite_calls can run over a cloned
    instruction list keyed by the ORIGINAL callee name (for its reloc map)."""

    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions
