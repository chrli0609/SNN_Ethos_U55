


# Constraints on Input and Output sizes of each layer

The output size of each layer must be a multiple of MIN_BLOCK_DEPTH, which is 8 for ethos u55 with configuration 256

The start address for each tensor/feature map must be 16-byte aligned if it uses NHCWB16 layout format

If the tensor/featuremap size is a multiple of 8 (but not 16), then the addresses for each NHCWB16 tensor/feature map will be padded to the next multiple of 16
