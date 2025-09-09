# Generate C layer and connectivity files
File structure for a model is:
model_name/
    - config_file.py
    - model_params/
        - fc_lif_layer_0_biases.npy
        - fc_lif_layer_0_weights.npy
        ...
        - fc_lif_layer_n_biases.npy
        - fc_lif_layer_n_weights.npy
    - test_patterns/
        - test_input_0.npy
        - test_target_0.npy
        ...
        - test_input_n.npy
        - test_target_n.npy


Generate files by running:
```
python3 main.py --model model_name
```


# Adding Operations other than Feed forward fully connected LIF
All the code for LIF is under gen_cms/fc_lif.py
That file is currently quite messy. Simplest way to add other operations is to create new file for it and import the npu_op_list generated



# Constraints on Input and Output sizes of each layer

The output size of each layer must be a multiple of MIN_BLOCK_DEPTH, which is 8 for ethos u55 with configuration 256

The start address for each tensor must be 16-byte aligned, since the addresses of the tensors are placed after each other,
if the output size is a multiple of 8 (and not 16), then the addresses for each tensor will be padded to the next multiple of 16