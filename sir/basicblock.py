class BasicBlock:
    def __init__(self, addr_content, pflag, instructions=[]):
        # The address of the start of this basic block
        self.addr_content = addr_content
        if addr_content != None:
            self.addr_content = addr_content.lower()
            # Calculate a numeric block address. For inlined blocks, addr_content may
            # be composite like "<callsite>_<origaddr>" (or nested), so combine hex
            # parts into a unique virtual address.
            parts = self.addr_content.split("_")
            addr = 0
            try:
                for part in parts:
                    addr = (addr << 32) | int(part, 16)
                self.addr = addr
            except ValueError:
                # Fallback: best-effort parse of the last component.
                self.addr = int(parts[-1], 16)
        # Instruction list
        self.instructions = instructions
        # Predecessor
        self.preds = []
        # Successors
        self.succs = []
        # Path flag
        self.pflag = pflag
        
    def add_pred(self, pred):
        if pred not in self.preds:
            self.preds.append(pred)

    def add_succ(self, succ):
        if succ not in self.succs:
            if succ == None:
                print("Add none successor???")
            self.succs.append(succ)
    
    # def is_isolated(self):
    #     return len(self.preds) == 0 and len(self.succs) == 0

    # # Initialize the basic bloci with entry address and branch flag
    # def init_block(self, addr_content, pflag):
    #     self.addr_content = addr_content
    #     self.addr = int(addr_content, base = 16)
    #     self.pflag = pflag
    
    # # Check if the basic block contains any instructions
    # def is_empty(self):
    #     return len(self.instructions) == 0

    # # Merge congtent with another basic block
    # def merge(self, another):
    #     # Append instruction list
    #     self.instructions = self.instructions + another.instructions
    #     # Erase old successor
    #     if another in self.succs:
    #         self.succs.remove(another)
    #     # Add new successor
    #     self.succs = self.succs + another.succs
        
    # # Erase the redundency in basic block
    # def erase_redundancy(self):
    #     #inst = self.instructions[0]
    #     #if inst.is_exit():
    #     #    # Empty the instruction list and just keep the exit instruction
    #     #    self.instructions = []
    #     #    self.instructions.append(inst)
    #     Idx = 0
    #     while Idx < len(self.instructions):
    #         Inst = self.instructions[Idx]
    #         if Inst.is_nop():
    #             # Remove NOP instructions
    #             self.instructions.remove(Inst)
    #         elif Inst.is_exit():
    #             Idx = Idx + 1
    #             # Erase the rest of instructions
    #             if Idx < len(self.instructions):
    #                 del self.instructions[Idx : len(self.instructions)] 
    #         else:
    #             Idx = Idx + 1
        
    def get_terminator(self):
        LastInst = self.instructions[-1]
        if LastInst.is_exit() or LastInst.is_branch() or LastInst.is_return() or LastInst.is_call():
            return LastInst
        else:
            return None

    def set_terminator(self, Inst):
        Inst = Inst.clone()
        self.instructions[-1] = Inst
        Inst.parent = self

    def get_branch_pair(self, Inst, Blocks):
        # Get the branch flag from branch instruction, i.e. P0 or !P0
        PFlag = Inst.get_branch_flag()
        if PFlag == None:
            return None
        
        if len(self.succs) != 2:
            print("Warning: More than/less than two successors in basic block")
            
        BB1 = None
        BB2 = None
        
        true_target = Inst.get_uses()[1].immediate_value
        false_target = Inst.get_uses()[2].immediate_value

        # Prefer local CFG edges to avoid ambiguity when multiple inlined copies
        # share the same original (non-virtual) addresses. Succs almost always
        # resolve both targets; only fall back to the full block list when they
        # don't (the eager succs+Blocks concat was O(#PBRAs x #blocks) — 14.7s
        # on cuDNN Ops so.75).
        for BB in self.succs:
            if BB1 is None and BB.addr == true_target:
                BB1 = BB
            if BB2 is None and BB.addr == false_target:
                BB2 = BB

        if BB1 is None or BB2 is None:
            for BB in Blocks:
                if BB in self.succs:
                    continue
                if BB1 is None and BB.addr == true_target:
                    BB1 = BB
                if BB2 is None and BB.addr == false_target:
                    BB2 = BB
                if BB1 is not None and BB2 is not None:
                    break

        if BB1 is None or BB2 is None:
            # Raise with context instead of returning None: the caller
            # unpacks the pair immediately, so None only surfaced as a
            # remote TypeError with the diagnostics lost.
            raise ValueError(
                f"Cannot resolve PBRA targets from block {self.addr:#x}: "
                f"true={true_target:#x} ({'found' if BB1 else 'missing'}), "
                f"false={false_target:#x} ({'found' if BB2 else 'missing'})")

        return BB1, BB2

    def dump(self):
        lines = []
        for inst in self.instructions:
            lines.extend(inst.dump())
        return lines

    def __str__(self):
        return "\n".join(self.dump())
        
