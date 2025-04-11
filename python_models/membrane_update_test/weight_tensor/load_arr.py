import numpy as np

data = np.load('main_split_1_flash')

np.set_printoptions(threshold = np.inf)



# Convert each element to a lowercase 2-digit hex string with '0x' prefix
hex_values = np.vectorize(lambda x: f"0x{x:02x}")(data)

# Flatten the array and join elements with commas
hex_string = ",".join(hex_values.flatten())


print(hex_string)
#print(data)
