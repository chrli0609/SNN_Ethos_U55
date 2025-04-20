#pragma once



#include <stdint.h>
#include <stddef.h>




#include "include/nn_ops.h"





int my_mem_update(
    float* in_spk,

    float* ln_beta,
    float* vth,
    float* v_mem,
    float* time_not_updated,

    float* out_spk
);


//int membrane_update(
//    size_t tensor_arena_size,
//    size_t input_layer_size,
//    size_t output_layer_size,
//
//    size_t num_time_steps_since_update,
//    size_t* beta_idx,
//
//
//    int8_t* in_spk,
//    int8_t* v_mem,
//    int8_t* decay,
//
//    const int8_t* weight_tensor_ptr,
//
//    int8_t* out_spk
//);

