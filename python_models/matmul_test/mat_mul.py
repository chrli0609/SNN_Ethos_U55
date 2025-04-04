import numpy as np




# Create matrix A (32 x 16), each row is [0, 1, 2, ..., 15]
A = np.tile(np.arange(1, 17), (32, 1))  # This creates A with values from 1 to 16

# Create matrix B (16 x 1), values from 15 down to 0
B = np.arange(15, -1, -1).reshape(16, 1)

# Perform matrix multiplication (A x B)
C = np.dot(A, B)


# Convert the result to uint8 (values will be clipped to the range [0, 255])
C = np.clip(C, 0, 255).astype(np.uint8)




# Print the result matrix C
print("A\n", A)
print("B\n", B)
print("C\n", C)

