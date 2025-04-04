import numpy as np

# Create matrix A (32 x 16), each row is [1, 2, 3, 4, 1, 2, 3, 4, ...]
A = np.tile(np.arange(1, 5), (32, 4)).T  # Repeat [1, 2, 3, 4] across 32 rows

# Create matrix B (16 x 1), repeating [1, 2, 3, 4] for 16 rows
#B = np.tile(np.arange(0, 4), 4)  # Repeat [1, 2, 3, 4] across 16 values, reshape to (16, 1)
B = np.tile(np.arange(0, 4), (1, 4*2)).T


# Print matrices A and B
print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)


# Perform matrix multiplication (A x B)
#C = np.dot(A, B)
C = np.dot(B, A)

# Convert the result to uint8 (values will be clipped to the range [0, 255])
C = np.clip(C, 0, 255).astype(np.uint8)

# Print the result matrix C
print("Matrix C:")
print(C)


print(A.shape)
print(B.shape)
print(C.shape)
