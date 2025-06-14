





#models = [
#        [784, 64, 64, 10],
#        [784, 48, 48, 48, 48, 10],
#        [784, 32, 32, 32, 32, 32, 32, 10],
#        [784, 16, 16, 16, 16, 16, 16, 16, 16, 10],
#]
models = [
        [784, 64, 64, 10],
        [784, 56, 56, 56, 56, 10],
        [784, 48, 48, 48, 48, 48, 48, 10],
        [784, 32, 32, 32, 32, 32, 32, 32, 32, 10],
]




def compute_num_model_params(layer_sizes_list):

    num_params = 0
    for i in range(len(layer_sizes_list)-1):

        num_params += layer_sizes_list[i] * layer_sizes_list[i+1]


    return num_params


def compute_memory_needed(layer_sizes_list):

    num_memory = 0

    for i in range(len(layer_sizes_list)-1):

        # For V_mem, tmp1, tmp2
        num_memory += 3*layer_sizes_list[i+1]

        # If is last layer, must also take out_spk_sum into account
        if i == len(layer_sizes_list)-2:
            num_memory += layer_sizes_list[i+1]


    return num_memory




for model in models:
    print("model:", model)
    print("\t- num_params:", compute_num_model_params(model), "\n\t- num_mem:", compute_memory_needed(model), "\n")
