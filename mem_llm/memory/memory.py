from Dynamic_Memory import Short_Term_Memory
from Static_Memory import Long_Term_Memory

dyanmic_memory = Short_Term_Memory()
static_memory = Long_Term_Memory()

# OrderDict
class Memory():
    def __init__(self, dyanmic_memory, static_memory, transit_threshold: int=0.9):
        self.dynamicMem = dyanmic_memory
        self.staticMem = static_memory
        self.transit_threshold = transit_threshold  # over the frequency, short->long

    def transit(self):
        pass

    

