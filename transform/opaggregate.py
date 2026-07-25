from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand
from collections import deque

from transform.opaggregate_patterns import PATTERN_TABLE

class OperAggregate(SaSSTransform):
    def apply(self, module):
        print("=== Start of Operator Aggregation Transformation ===")

        total_insert = 0
        total_remove = 0
        total_patterns = 0

        for func in module.functions:

            self.insert_insts = {}
            self.remove_insts = set()
            self.handled_insts = set()
            self.known_reg_pairs = {}
            self.processed_patterns = []

            seed_patterns = self.find_seed_patterns(func)

            self.track_patterns(seed_patterns)
            
            patterns = len(self.processed_patterns)
            total_patterns += patterns
            
            self.fix_register_dependencies()

            self.apply_changes(func)

            print(f"Function {func.name}:")
            print(f"\tOperAggregate Patterns Found: {patterns}")
            print(f"\tOperAggregate Insert Instructions: {len(self.insert_insts)}")
            print(f"\tOperAggregate Remove Instructions: {len(self.remove_insts)}")

            total_insert += len(self.insert_insts)
            total_remove += len(self.remove_insts)

        print(f"Total OperAggregate Patterns Found: {total_patterns}")
        print(f"Total OperAggregate Insert Instructions: {total_insert}")
        print(f"Total OperAggregate Remove Instructions: {total_remove}")
        self.total_patterns = total_patterns

        print("=== End of Operator Aggregation Transformation ===")
        
    # def GetLDG64Instructions(self, func):
    #     LDG64Insts = []
    #     for bb in func.blocks:
    #         for inst in bb.instructions:
    #             if len(inst.opcodes) > 0 and inst.opcodes[0] == "LDG" and "64" in inst.opcodes:
    #                 LDG64Insts.append(inst)

    #     return LDG64Insts
            
    # def GetPack64Instructions(self, func):
    #     Pack64Insts = []
    #     for bb in func.blocks:
    #         for inst in bb.instructions:
    #             if len(inst.opcodes) > 0 and inst.opcodes[0] == "PACK64":
    #                 Pack64Insts.append(inst)

    #     return Pack64Insts
    
    def iterate_opcodes(self, opcode_inst_dict, opcode_seq, order_index):
        """
        ai.
        Yield instruction tuples (i0, i1, ..., in) where i_k has opcode opcode_seq[k],
        and their program order is strictly increasing within the basic block.
        """
        inst_lists = []
        for op in opcode_seq:
            lst = opcode_inst_dict.get(op)
            if not lst:
                return
            inst_lists.append(lst)

        def dfs(k, prev_idx, acc):
            if k == len(inst_lists):
                yield tuple(acc)
                return
            for inst in inst_lists[k]:
                idx = order_index.get(inst)
                if idx is None:
                    continue
                if prev_idx is None or idx > prev_idx:
                    acc.append(inst)
                    yield from dfs(k + 1, idx, acc)
                    acc.pop()

        yield from dfs(0, None, [])

    
    def find_seed_patterns(self, func):
        seed_patterns = []
        
        opcode_patterns = []
        
        for pattern_key in PATTERN_TABLE.keys():
            # phi and mov not matched because they can be trivially paired
            if not pattern_key or pattern_key[0] == "PHI" or pattern_key[0] == "MOV":
                continue
            # # ldg not matched because ldg.64 can be for float64 or int64
            # # for now, just avoid this complexity
            # if patternKey[0] == "LDG":
            #     continue
            opcode_patterns.append(pattern_key)
    
        for bb in func.blocks:
            # Cache instruction by opcode for quick lookup
            opcode_inst_dict = {}
            order_index = {}
            for idx, inst in enumerate(bb.instructions):
                if len(inst.opcodes) == 0:
                    continue
                order_index[inst] = idx
                opcode_inst_dict.setdefault(inst.opcodes[0], []).append(inst)
                
            print(f"Processing basic block...")
            
            # Find seed instruction pairs
            for opcode_seq in opcode_patterns:
                # quick skip if any opcode is absent in this block
                if any(op not in opcode_inst_dict for op in opcode_seq):
                    continue

                for inst_tuple in self.iterate_opcodes(opcode_inst_dict, opcode_seq, order_index):
                    # print(f"Trying instruction tuple: ({', '.join(f'<{inst}>' for inst in InstTuple)})")
                    
                    status, _, _ = self.match_patterns(list(inst_tuple), check_only=True)
                    if status:
                        seed_patterns.append(list(inst_tuple))
                        print(f"Found seed instruction tuple: ({', '.join(f'<{inst}>' for inst in inst_tuple)})")

        return seed_patterns
    
    

    def track_patterns(self, seed_patterns):
        # reg3:reg2 -> (reg2,reg3) dictionary
        # all regs are SSA so no conflict
        queue = deque()

        for seed_pattern in seed_patterns:
            queue.append(seed_pattern)

        while len(queue) > 0:
            pattern_insts = queue.popleft()
            
            print(f"({', '.join(f'<{inst}>' for inst in pattern_insts)})")
                    
            if len(pattern_insts) == 0:
                continue

            # Identify pattern, get replace instruction, next reg pairs candidates
            status, merge_insts, reg_pairs = self.match_patterns(pattern_insts)
                
                
            # Skip if any inst in pattern_insts is already handled
            if any(inst in self.handled_insts for inst in pattern_insts):
                # If not all of them being handled, print a warning
                if not all(inst in self.handled_insts for inst in pattern_insts):
                    print(f"\tWarning: not all instructions handled for pattern {pattern_insts}")
                print("=> Already handled, skipping\n")
                continue
            
            # Check match status
            if not status:
                print(f"\tChain ended for this pair")
                print("=> No pattern match\n")
                continue
                
            # Mark all instructions in pattern as handled
            for inst in pattern_insts:
                self.handled_insts.add(inst)

            end_of_chain = len(reg_pairs) == 0
            
            # Convert operands to instructions
            next_patterns = []
            for reg_pair in reg_pairs:
                next_patterns.extend(self.get_matching_pairs(reg_pair))
                
            
                # For each reg pair, make sure all def instructions are covered
                # Reg1Defs = set()
                # Reg1Defs.add((RegPair[0][0].Parent, RegPair[0][0]))
                    
                # Reg2Defs = set()
                # Reg2Defs.add((RegPair[1][0].Parent, RegPair[1][0]))

                # if not self.all_defs_covered(MatchingPairs, Reg1Defs, Reg2Defs):
                #     InsertInsts[Inst1].append(self.CreatePack64(Inst1, RegPair[0], RegPair[1]))
                #     EndOfChain = True
                #     print(f"\tPack64 inserted for operand ({RegPair[0][1]}, {RegPair[1][1]})")
                
            # Add to processed patterns
            self.processed_patterns.append([pattern_insts, merge_insts])

            if not end_of_chain:
                queue.extend(next_patterns)
                for next_pattern in next_patterns:
                    print(f"\tNext pair(use->def): (", end="")
                    for next_pattern_inst in next_pattern:
                        print(f"{next_pattern_inst},", end=" ")
                    print(")")
            else:
                print(f"\tChain ended for this pair")

            print(f"=> {merge_insts}\n")

    # check_only: won't generate new instructions
    def match_patterns(self, pattern_insts, check_only=False, resolve_define=True):

        def is_variable(token):
            prefixes = ("reg", "pred", "const", "op", "imm")
            return any(token.startswith(prefix) for prefix in prefixes)

        def variable_type_matches(var_name, value):
            prefix_end = 0
            for idx, ch in enumerate(var_name):
                if ch.isalpha():
                    prefix_end = idx + 1
                else:
                    break
            prefix = var_name[:prefix_end]

            if prefix == "reg":
                return isinstance(value, Operand) and value.is_reg
            if prefix == "pred":
                return isinstance(value, Operand) and value.is_predicate_reg
            if prefix == "const":
                return isinstance(value, Operand) and value.is_const_mem
            if prefix == "imm":
                return isinstance(value, Operand) and value.is_immediate
            if prefix == "op":
                return True
            return True
        
        def match_opcodes(pattern_opcodes, inst_opcodes, variables):
            if len(pattern_opcodes) > len(inst_opcodes):
                return False
            
            for pattern_opcode, opcode in zip(pattern_opcodes, inst_opcodes):
                if "op" in pattern_opcode:
                    if pattern_opcode in variables and variables[pattern_opcode] != opcode:
                        return False
                    else:
                        variables[pattern_opcode] = opcode
                else:
                    if pattern_opcode != opcode:
                        return False
            return True
        
        def match_operands(pattern_operands, inst_operands, variables):
            if len(pattern_operands) == 1 and "[*]" in pattern_operands[0]:
                pattern_operands = [pattern_operands[0].replace("[*]", str(i)) for i in range(len(inst_operands))]
                pack_length = len(inst_operands)
                if "pack_length" in variables:
                    # This prevents two phi having different number of operands to be matched.
                    if variables["pack_length"] != pack_length:
                        return False
                else:
                    variables["pack_length"] = pack_length

            if len(pattern_operands) > len(inst_operands):
                return False
            
            for pattern_operand, operand in zip(pattern_operands, inst_operands):
                if is_variable(pattern_operand):
                    if not variable_type_matches(pattern_operand, operand):
                        return False
                    if pattern_operand in variables and variables[pattern_operand].name != operand.name:
                        return False
                    else:
                        variables[pattern_operand] = operand
                else:
                    if pattern_operand != operand.name:
                        return False
            return True
        
        def gen_array(pattern_array, variables, operand_parser=Operand.parse):
            array = []
            for item in pattern_array:
                if is_variable(item):
                    if item in variables:
                        array.append(variables[item])
                    else:
                        array.append(operand_parser(item))
                else:
                    array.append(operand_parser(item))
            return array
        
        def resolve_defined(pattern_defined, variables):
            def find_operand_in_def_inst(reg_op_in_use_inst):
                if len(reg_op_in_use_inst.defining_insts) > 1:
                    raise Exception("Multiple defining instructions found for operand")
                
                for def_inst in reg_op_in_use_inst.defining_insts:
                    for def_op in def_inst.get_defs():
                        if def_op.reg == reg_op_in_use_inst.reg:
                            return def_op
                return None
            
            if len(pattern_defined) == 0:
                return True
            
            for pattern_def in pattern_defined:
                regs = pattern_def.split(":")
                if len(regs) != 2:
                    raise Exception("Defined reg pair must have two regs")
                
                reg2, reg1 = regs
                
                # Find the corresponding operands in defining instructions
                if reg1 in variables:
                    reg1_def_op = find_operand_in_def_inst(variables[reg1])
                if reg2 in variables:
                    reg2_def_op = find_operand_in_def_inst(variables[reg2])
                    
                
                if reg1 in variables and reg1_def_op in self.known_reg_pairs:
                    variables[reg2] = self.known_reg_pairs[reg1_def_op][0]
                elif reg2 in variables and reg2_def_op in self.known_reg_pairs:
                    variables[reg1] = self.known_reg_pairs[reg2_def_op][1]
                else:
                    return False
                
            return True
        
        def update_known_reg_pairs(out_inst_pattern, variables):
            for key, pattern_operands in out_inst_pattern.items():
                if key != "def" and key != "use":
                    continue
                
                if len(pattern_operands) == 1 and "[*]" in pattern_operands[0]:
                    operands_length = variables["pack_length"]
                    pattern_operands = [pattern_operands[0].replace("[*]", str(i)) for i in range(operands_length)]

                for pattern_operand in pattern_operands:
                    p = pattern_operand.split(":")
                    for i in range(len(p)):
                        if is_variable(p[i]):
                            var = variables[p[i]]
                        else:
                            var = p[i]
                        p[i] = var

                    if len(p) == 2:
                        print(f"\tKnown reg pair: ({p[0]}, {p[1]})")
                        self.known_reg_pairs[p[0]] = (p[0], p[1])
                        self.known_reg_pairs[p[1]] = (p[0], p[1])
                        
        def gen_opcodes(pattern_array, variables):
            opcodes = []
            for p in pattern_array:
                if is_variable(p):
                    opcodes.append(variables[p])
                else:
                    opcodes.append(p)
            return opcodes
        
        def gen_next_array(pattern_array, variables):
            if len(pattern_array) == 1 and "[*]" in pattern_array[0][0]:
                operands_length = variables["pack_length"]
                pattern_array = [(pattern_array[0][0].replace("[*]", str(i)), pattern_array[0][1].replace("[*]", str(i))) for i in range(operands_length)]

            reg_pairs = []
            for pattern_pair in pattern_array:
                if len(pattern_pair) != 2:
                    raise Exception("Next reg pair must have two regs")
                
                reg1, reg2 = pattern_pair
                if is_variable(reg1):
                    reg1_op = variables[reg1]
                else:
                    reg1_op = Operand.parse(reg1)
                    
                if is_variable(reg2):
                    reg2_op = variables[reg2]
                else:
                    reg2_op = Operand.parse(reg2)
                
                reg_pairs.append((reg1_op, reg2_op))
                
            return reg_pairs

        def gen_operands(pattern_operands, variables):
            if len(pattern_operands) == 1 and "[*]" in pattern_operands[0]:
                operands_length = variables["pack_length"]
                pattern_operands = [pattern_operands[0].replace("[*]", str(i)) for i in range(operands_length)]

            array = []
            for pattern_operand in pattern_operands:
                p = pattern_operand.split(":")
                for i in range(len(p)):
                    if is_variable(p[i]):
                        name = variables[p[i]].name
                    else:
                        name = p[i]
                    p[i] = name
                array.append(Operand.parse(":".join(p)))

            return array
        
        def handle_rz(use_operands, parent_inst, out_insts):
            for use_op in use_operands:
                if "RZ" in use_op.name and ":" in use_op.name:
                    regs = use_op.name.split(":")

                    operands = [use_op.clone(), Operand.parse(regs[0]), Operand.parse(regs[1])]

                    pack64_inst = Instruction(
                        id=f"{parent_inst.id}_pack64_{use_op.name}",
                        opcodes=["PACK64"],
                        operands=operands,
                        parentBB=parent_inst.parent
                    )
                    out_insts.append(pack64_inst)
                    print(f"\tPack64 inserted for operand {use_op}")

        # Match pattern
        opcodes_pattern = tuple(inst.opcodes[0] for inst in pattern_insts)
        if opcodes_pattern not in PATTERN_TABLE:
            return False, None, None
        current_patterns = PATTERN_TABLE[opcodes_pattern]
        
        for pattern in current_patterns:
            in_insts_pattern = pattern["in"]
            out_insts_pattern = pattern["out"]
            
            variables = {"RZ": Operand.parse("RZ"), "PT": Operand.parse("PT")}
            matched = True
            
            if len(in_insts_pattern) != len(pattern_insts):
                continue

            for idx, in_inst_pattern in enumerate(in_insts_pattern):
                in_inst = pattern_insts[idx]

                # Match opcodes
                pattern_in_inst_opcodes = in_inst_pattern["opcodes"]
                if not match_opcodes(pattern_in_inst_opcodes, in_inst.opcodes, variables):
                    matched = False
                    break

                # Match def operands
                pattern_in_inst_defs = in_inst_pattern["def"]
                if not match_operands(pattern_in_inst_defs, in_inst.get_defs(), variables):
                    matched = False
                    break
                
                # Match use operands
                pattern_in_inst_uses = in_inst_pattern["use"]
                if not match_operands(pattern_in_inst_uses, in_inst.get_uses(), variables):
                    matched = False
                    break
                
                # Resolve defined reg pairs 
                if resolve_define:
                    pattern_defined = pattern.get("defined", [])
                    if not resolve_defined(pattern_defined, variables):
                        matched = False
                        break

            if not matched:
                continue
            

            # Skip early if check_only
            if check_only:
                return True, None, None

            # Generate output instructions
            out_insts = []
            for out_inst_pattern in out_insts_pattern:
                
                update_known_reg_pairs(out_inst_pattern, variables)
                
                reg_pairs = gen_next_array(pattern.get("next", []), variables)

                inst_id = f"{in_inst.id}_x64"
                opcodes = gen_opcodes(out_inst_pattern["opcodes"], variables)
                defs = gen_operands(out_inst_pattern["def"], variables)
                uses = gen_operands(out_inst_pattern["use"], variables)
                operands = defs + uses
                parent = in_inst.parent
                
                out_inst = Instruction(
                    id=inst_id,
                    opcodes=opcodes,
                    operands=operands,
                    parentBB=parent
                )
                
                # Handle RZ. For example, if an use operand is RZ:R2, 
                # insert PACK64 RZ:R2=RZ,R2 before outInst 
                handle_rz(uses, out_inst, out_insts)
                
                out_insts.append(out_inst)

                return True, out_insts, reg_pairs

        return False, None, None

    def get_matching_pairs(self, reg_pairs):
        reg1_op = reg_pairs[0]
        reg2_op = reg_pairs[1]
        reg1_insts = reg1_op.defining_insts
        reg2_insts = reg2_op.defining_insts
        matching_pairs = []
        
        # Pattern could be all combinations: R1 only, R2 only, R1 and R2
        if len(reg1_insts) == 0:
            match_r1 = False
            match_r2 = True
            match_r1_and_r2 = False
        elif len(reg2_insts) == 0:
            match_r2 = False
            match_r1 = True
            match_r1_and_r2 = False
        elif reg1_insts == reg2_insts:
            match_r1 = True
            match_r2 = False
            match_r1_and_r2 = False
        else:
            match_r1 = True
            match_r2 = True
            match_r1_and_r2 = True
        
        if match_r1_and_r2:
            for reg1_def_inst in reg1_insts:
                for reg2_def_inst in reg2_insts:
                    if self.control_equivalent(reg1_def_inst, reg2_def_inst):
                        status, _, _ = self.match_patterns([reg1_def_inst, reg2_def_inst], check_only=True, resolve_define=False)
                        if status:
                            matching_pairs.append((reg1_def_inst, reg2_def_inst))
        if match_r2:
            for reg_def_inst in reg2_insts:
                status, _, _ = self.match_patterns([reg_def_inst], check_only=True, resolve_define=False)
                if status:
                    matching_pairs.append((reg_def_inst,))
        if match_r1:
            for reg_def_inst in reg1_insts:
                status, _, _ = self.match_patterns([reg_def_inst], check_only=True, resolve_define=False)
                if status:
                    matching_pairs.append((reg_def_inst,))
        
        return matching_pairs
    
    def control_equivalent(self, inst1, inst2):
        # TODO: Inst1.dominate(Inst2) and Inst2.postdominate(Inst1)
        return True
    
    def all_defs_covered(self, matching_pairs, reg1_defs, reg2_defs):
        covered_reg1 = set()
        covered_reg2 = set()
        for reg1_def, reg2_def in matching_pairs:
            covered_reg1.add(reg1_def)
            covered_reg2.add(reg2_def)

        return covered_reg1 == reg1_defs and covered_reg2 == reg2_defs
    
    def fix_register_dependencies(self):

        for pattern in self.processed_patterns:
            in_insts = pattern[0]
            out_insts = pattern[1]

            # go over user instruction of every in_insts entry to make sure every user is converted
            all_converted = True
            if len(in_insts) > 1 or (len(in_insts) == 1 and len(in_insts[0].get_defs()) > 1):
                for inst in in_insts:
                    for user in inst.users.values():
                        for user_inst, _ in user:
                            if user_inst not in self.handled_insts:
                                all_converted = False

            # Remove pattern instructions
            for inst in in_insts:
                self.remove_insts.add(inst)
                
            # Insert output instructions after the last pattern instruction    
            self.insert_insts[in_insts[-1]] = out_insts
            
            # If not all users converted, insert UNPACK64
            if not all_converted:
                print(f"\tNot all users converted, inserting UNPACK64 after {in_insts[-1]}")
                self.insert_insts[in_insts[-1]].extend(self.create_unpack64(out_insts))
            
    def create_unpack64(self, out_insts):
        unpack64_insts = []
        
        for out_inst in out_insts:
            for def_op in out_inst.get_defs():
                if ":" not in def_op.name:
                    continue
                
                regs = def_op.name.split(":")
                operands = [Operand.parse(regs[0]), Operand.parse(regs[1]), def_op.clone()]
                
                unpack64_inst = Instruction(
                    id=f"{out_inst.id}_unpack64_{operands[0].name}_{operands[1].name}",
                    opcodes=["UNPACK64"],
                    operands=operands,
                    parentBB=out_inst.parent
                )
                unpack64_insts.append(unpack64_inst)
                print(f"\tUnpack64 inserted for operand ({operands[0].name}, {operands[1].name}) from {def_op.name}")

        return unpack64_insts

    def apply_changes(self, func):

        for bb in func.blocks:

            new_insts = []

            for inst in bb.instructions:

                if inst in self.insert_insts:
                    insert_insts = self.insert_insts[inst]
                    new_insts.extend(insert_insts)

                
                if inst not in self.remove_insts:
                    new_insts.append(inst)

            bb.instructions = new_insts
