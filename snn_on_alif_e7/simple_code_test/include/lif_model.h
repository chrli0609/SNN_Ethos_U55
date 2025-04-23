#pragma once



//#include <stdint.h>
#include <stddef.h>


//#include "include/nn_ops.h"
#include "include/nn_data_structure.h" //for NN_Model struct









int my_mem_update(
    NN_Model* mlp_model,
    float* in_spk,

    float* ln_beta,
    float* vth,
    float* v_mem,
    float* time_not_updated,

    float* out_spk
);
