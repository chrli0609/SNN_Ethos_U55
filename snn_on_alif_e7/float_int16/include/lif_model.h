#pragma once



#include <stdint.h>
#include <stddef.h>




#include "include/nn_ops.h"





int my_mem_update(
    float* in_spk,
    float* v_mem,
    float* decay,
    //int16_t* v_mem,
    //size_t v_mem_tensor_size,
    //int16_t* decay,
    //size_t decay_tensor_size,

    //relative addressing
    //size_t in_spk_rel_addr,
    //size_t out_spk_rel_addr,





    //int16_t* v_mem_new,
    //size_t v_mem_new_tensor_size,
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
//    int16_tt* in_spk,
//    int8_t* v_mem,
//    int16_t* decay,
//
//    const int8_t* weight_tensor_ptr,
//
//    int16_t* out_spk
//);

