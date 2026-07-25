class Function:
    
    #def __init__(self, parser):
    #    self.name = ""
    #    self.ParseFunction(parser)

    def __init__(self, name):
        self.name = name
        self.instructions = []
        self.blocks = []
        self.headerflags = None
        self.args = []
        self.arg_map = {}
        self.block_map = {}
        
    # Get the arguments for current function
    def get_args(self, lifter):
        args = []
        arg_map = {}
        # Collect the arguments
        for bb in self.blocks:
            for inst in bb.instructions:
                for operand in inst.operands:
                    if operand.is_arg:
                        name = operand.get_ir_name(lifter)
                        arg_map[name] = operand
                        
        # Keep one operand per argument
        unique = {}
        for operand in arg_map.values():
            unique.setdefault(operand.offset, operand)

        # Sort the map
        args = sorted(unique.values(), key=lambda x: x.offset)

        return args

    # Create control flow graph
    def dump_cfg(self):
        for bb in self.blocks:
            succs = []
            for succ in bb.succs:
                succs.append(succ.addr_content)
            print("BB: ", bb.addr_content, succs)

    def __str__(self):
        lines = [f"\t\tFunction : {self.name}"]
        if self.blocks:
            for bb in self.blocks:
                lines.extend(bb.dump())
        else:
            for inst in self.instructions:
                lines.extend(inst.dump())
        return "\n".join(lines)

    def dump(self):
        lines = [f"\t\tFunction : {self.name}"]
        if self.headerflags:
            lines.append(f"\t.headerflags\t@\"{self.headerflags}\"")
        if self.blocks:
            for bb in self.blocks:
                lines.extend(bb.dump())
        else:
            for inst in self.instructions:
                lines.extend(inst.dump())
        return lines

    # Build the map between basic block and its IR version
    def build_bb_to_ir_map(self, ir_func, block_map):
        is_entry = True
        for bb in self.blocks:
            if is_entry:
                bb_name = "EntryBB_" + bb.addr_content
            else:
                bb_name = "BB_" + bb.addr_content

            # Create the basic block
            ir_block = ir_func.append_basic_block(bb_name)
            # Register IR block
            block_map[bb] = ir_block
            
            is_entry = False
