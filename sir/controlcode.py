class PresetCtlCodeException(Exception):
    pass

class ControlCode:
    def __init__(self, content, WaitB, ReadB, WrtB, Yield, Stall):
        self.content = content
        self.WaitB = WaitB
        self.ReadB = ReadB
        self.WrtB = WrtB
        self.Yield = Yield
        self.Stall = Stall
        
    def dump(self):
        print("wait {} read {} wrt {} yield {} stall {}".format(self.WaitB, self.ReadB, self.WrtB, self.Yield, self.Stall))
