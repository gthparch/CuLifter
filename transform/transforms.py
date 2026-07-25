from __future__ import annotations

from typing import List, Sequence

from transform.transform import SaSSTransform


class Transforms:
    def __init__(self, passes: Sequence[SaSSTransform], *, verbose: bool = False) -> None:
        self.passes: List[SaSSTransform] = list(passes)
        self.verbose = verbose

    def apply(self, module) -> None:
        if self.verbose:
            self._dump_module(module, "Initial module")

        import os, sys, time
        from transform.defuse_analysis import DefUseAnalysis
        log_slow = os.environ.get("SLIFTER_PASS_TIMING") == "1"
        # A rebuild is skippable iff every pass since the last DefUseAnalysis
        # declared `self.changed = False` (no def/use mutations). Passes that
        # don't report (attr None/absent) conservatively force the rebuild.
        # DefUse rebuilds are the #1 lift cost on large modules (105s of 341s
        # on cuDNN Ops so.75) and IntDivRecognize usually matches nothing.
        defuse_fresh = False
        for tranpass in self.passes:
            if isinstance(tranpass, DefUseAnalysis) and defuse_fresh:
                continue
            t0 = time.monotonic()
            tranpass.apply(module)
            if isinstance(tranpass, DefUseAnalysis):
                defuse_fresh = True
            elif getattr(tranpass, "changed", None) is not False:
                defuse_fresh = False
            dt = time.monotonic() - t0
            # Surface any pass that takes > 5s — common O(n^2)+ culprits
            # (SSA, DefUse, simplify) on large kernels show up here.
            if log_slow and dt > 5.0:
                func_names = ",".join(f.name[:40] for f in module.functions[:1])
                print(f"[pass-timing] {tranpass.name:40s} {dt:7.2f}s on {func_names}",
                      file=sys.stderr, flush=True)
            if self.verbose:
                self._dump_module(module, f"After {tranpass.name}")

    def _dump_module(self, module, title: str) -> None:
        print(title)
        for func in module.functions:
            print(f"Function: {func.name}")
            for block in func.blocks:
                preds = ",".join(bb.addr_content for bb in block.preds)
                succs = ",".join(bb.addr_content for bb in block.succs)
                print(f"  Block: {block.addr_content} from: [{preds}] to: [{succs}]")
                for inst in block.instructions:
                    print(f"    {inst.id}    {inst}")
            print("")
