


# Constraints on Input and Output sizes of each layer

The output size of each layer must be a multiple of MIN_BLOCK_DEPTH, which is 8 for ethos u55 with configuration 256

The start address for each tensor must be 16-byte aligned, since the addresses of the tensors are placed after each other,
if the output size is a multiple of 8 (and not 16), then the addresses for each tensor will be padded to the next multiple of 16
