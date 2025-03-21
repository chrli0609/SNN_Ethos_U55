import numpy as np

import sys

flash_file = sys.argv[1]

data = np.load(flash_file)

np.set_printoptions(formatter={'int': lambda x: f"0x{x:02x}"}, threshold=np.inf)
print("Printing as hex!!")
print(repr(data))
