from sir.function import Function
from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.controlcode import ControlCode
from sir.operand import Operand
from sir.operand import InvalidOperandException

from parse.parser_base import SaSSParserBase
from parse.parser_base import UnmatchedControlCode

SM52_CTLCODE_LEN = 3

class SaSSParser_SM52(SaSSParserBase):
    def __init__(self, isa, file, elf):
        super().__init__(isa, file, elf)
        self.CtlCodes = []


    # Parse the function body from file lines
    def parse_func_body(self, line, Insts, CurrFunc):
        # Process function body 
        items = line.split('*/')
        if len(items) == 2:
            # Parse the control code
            self.parse_control_code(items[0], self.CtlCodes)
        elif len(items) == 3:    
            # Retrieve instruction ID
            inst_id = self.get_inst_num(items[0])
            # Retrieve instruction opcode
            inst_opcode, pflag, rest_content = self.get_inst_opcode(items[1])
            rest_content = rest_content.replace(" ", "")

            # Retrieve instruction operands
            inst_ops = self.get_inst_operands(rest_content)
            
            encoding = self._extract_inst_encoding(line)
            # Create instruction
            inst = self.parse_instruction(
                inst_id,
                inst_opcode,
                pflag,
                inst_ops,
                rest_content,
                CurrFunc,
                encoding,
            )

            # Add control code
            if len(self.CtlCodes) > 0:
                CtlCode = self.CtlCodes[0]
                inst.set_ctl_code(CtlCode)
                # Remove the control code from temprory storage
                self.CtlCodes.remove(CtlCode)
            # else:
            #     raise UnmatchedControlCode
            
            # Add instruction into list
            Insts.append(inst)
            
    
    # Parse control code 
    def parse_control_code(self, Content, ControlCodes):
        Content = Content.replace("/*", "")
        Content = Content.replace(" ", "")

        # This parsing process is correspoinding to Maxwell architecture, which has
        # 3 17-bits sections for control and 3 4-bits sections for reuse

        # Extract control sections
        Sec1 = Content[2 : 8]
        Sec2 = Content[8 : 13]
        Sec3 = Content[13 : ]
        Sec1Num = int("0x" + Sec1, 16) & int("0x7fffc", 16)
        Sec2Num = int("0x" + Sec2, 16) & int("0x3fffe", 16)
        Sec3Num = int("0x" + Sec3, 16) & int("0x1ffff", 16)

        Stall = (Sec3Num & 15) >> 0
        Yield = (Sec3Num & 16) >> 4
        WrtB  = (Sec3Num & 224) >> 5
        ReadB = (Sec3Num & 1792) >> 8
        WaitB = (Sec3Num & 129024) >> 11
        ControlCodes.append(ControlCode(Content, WaitB, ReadB, WrtB, Yield, Stall))
        
        Stall = (Sec2Num & 15) >> 0
        Yield = (Sec2Num & 16) >> 4
        WrtB  = (Sec2Num & 224) >> 5
        ReadB = (Sec2Num & 1792) >> 8
        WaitB = (Sec2Num & 129024) >> 11
        ControlCodes.append(ControlCode(Content, WaitB, ReadB, WrtB, Yield, Stall))
        
        Stall = (Sec1Num & 15) >> 0
        Yield = (Sec1Num & 16) >> 4
        WrtB  = (Sec1Num & 224) >> 5
        ReadB = (Sec1Num & 1792) >> 8
        WaitB = (Sec1Num & 129024) >> 11
        ControlCodes.append(ControlCode(Content, WaitB, ReadB, WrtB, Yield, Stall))

        return ControlCodes
