from sir.function import Function
from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.controlcode import ControlCode
from sir.operand import Operand
from sir.operand import InvalidOperandException

from parse.parser_base import SaSSParserBase
from parse.parser_base import UnmatchedControlCode

SM75_CTLCODE_LEN = 1

class SaSSParser_SM75(SaSSParserBase):
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

            if len(Insts) <= 0:
                raise UnmatchedControlCode
            else:
                Idx = len(Insts) - 1
                CtlCode = self.CtlCodes[0]
                Insts[Idx].set_ctl_code(CtlCode)
                
                # Remove the control code from temprory storage
                self.CtlCodes.remove(CtlCode)
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

            # Add instruction into list
            Insts.append(inst)
            
    # Parse control code 
    def parse_control_code(self, Content, ControlCodes):
        Content = Content.replace("/*", "")
        Content = Content.replace(" ", "")

        # This parsing process is correspoinding to Volta and Turing architecture, which 
        # has one 17-bits sections in the top part of 128-bits instruction

        # Extract control sections
        Sec = Content[2 : 8]

        SecNum = int("0x" + Sec, 16) & int("0xffffe", 16)
        
        Stall = (SecNum & 15) >> 0
        Yield = (SecNum & 16) >> 4
        WrtB  = (SecNum & 224) >> 5
        ReadB = (SecNum & 1792) >> 8
        WaitB = (SecNum & 129024) >> 11
        ControlCodes.append(ControlCode(Content, WaitB, ReadB, WrtB, Yield, Stall))
        
        return ControlCodes
