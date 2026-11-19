from parse.parser_sm52 import SaSSParser_SM52
from parse.parser_sm35 import SaSSParser_SM35
from parse.parser_sm75 import SaSSParser_SM75
from parse.parser_nvbit_sm75 import SaSSParser_NVBit_SM75
from parse.parser_nvdisasm_sm75 import SaSSParser_NVDisasm_SM75

class InvalidISAException(Exception):
    pass

class SaSSParser:
    def __init__(self, file, elf):
        self.file = file
        self.is_nvbit = False
        self.is_nvdisasm = False
        self.elf = elf

        isa = self.get_isa(file)
        self.isa = isa

        if self.is_nvbit:
            self.parser = SaSSParser_NVBit_SM75(isa, file, elf)
        elif self.is_nvdisasm:
            self.parser = SaSSParser_NVDisasm_SM75(isa, file, elf)
        elif isa == "sm_52":
            self.parser = SaSSParser_SM52(isa, file, elf)
        elif isa == "sm_35":
            self.parser = SaSSParser_SM35(isa, file, elf)
        elif isa == "sm_75":
            self.parser = SaSSParser_SM75(isa, file, elf)
        elif isa == "sm_70" or isa == "sm_70a":
            self.parser = SaSSParser_SM75(isa, file, elf)
        elif isa == "sm_80" or isa == "sm_80a":
            self.parser = SaSSParser_SM75(isa, file, elf)
        elif isa == "sm_86" or isa == "sm_86a":
            self.parser = SaSSParser_SM75(isa, file, elf)
        elif isa == "sm_89" or isa == "sm_89a":
            self.parser = SaSSParser_SM75(isa, file, elf)
        elif isa == "sm_90" or isa == "sm_90a":
            self.parser = SaSSParser_SM75(isa, file, elf)
        elif isa == "sm_100" or isa == "sm_100a":
            self.parser = SaSSParser_SM75(isa, file, elf)
        elif isa == "sm_120" or isa == "sm_120a":
            self.parser = SaSSParser_SM75(isa, file, elf)
        else:
            raise InvalidISAException

        # Make ELF available to downstream parsing stages (CFG creation, etc).
        self.parser.elf = elf
            
    # Retrieve ISA definition
    def get_isa(self, file):
        for line_num, line in enumerate(file.split('\n')):
            if "code for sm" in line :
                items = line.split("code for ")
                return items[1]

        if "NVBit (NVidia Binary Instrumentation Tool" in file or "inspecting " in file:
            self.is_nvbit = True
            return "sm_75"

        if ".nv.info" in file and ".text." in file:
            self.is_nvdisasm = True
            return self.get_nvdisasm_isa(file)

        # Fallback: check for "arch = sm_XXX" in fatbin elf headers
        # (cuDNN cubins use sm_90a and extracted sections may lack
        #  the "code for sm_XXX" line)
        for line in file.split('\n'):
            if "arch = sm_" in line:
                parts = line.split("arch = ")
                if len(parts) >= 2:
                    return parts[1].strip()

        raise InvalidISAException

    def get_nvdisasm_isa(self, file):
        # New nvdisasm dumps use .target instead of .headerflags. Keep the
        # accelerated suffix because it determines the legal Blackwell PTX.
        for line in file.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == ".target" and fields[1] in (
                    "sm_100", "sm_100a", "sm_120", "sm_120a"):
                return fields[1]
        for line in file.split('\n'):
            if "EF_CUDA_SM" in line:
                parts = line.split("EF_CUDA_SM", 1)
                if len(parts) == 2:
                    digits = ""
                    for ch in parts[1]:
                        if ch.isdigit():
                            digits += ch
                        else:
                            break
                    if digits:
                        return f"sm_{digits}"
        return "sm_75"
        
    # Parse the SaSS text file
    def apply(self):
        return self.parser.apply()
    
