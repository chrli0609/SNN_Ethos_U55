#pragma once

#include "../include/nn_data_structure.h"







// Include Test Patterns
//#include "test_patterns/pattern_0.h"
//#include "test_patterns/pattern_1.h"

// Set NNLayer Tensor Indices
#define IN_SPK_TENSOR_IDX               0
#define BIAS_TENSOR_IDX                 1
#define WEGIHTS_TENSOR_IDX              2
#define V_MEM_QUANT_IDX                 3
#define TIME_NOT_UPDATED_QUANT_IDX      4
#define UPDATE_NXT_LAYER_IDX            5
#define OUT_SPK_TENSOR_IDX              6

// Only exists for last layer
#define OUT_SPK_SUM_TENSOR_IDX          7



NNLayer* FC_LIF_Layer_Init(

    // Const Tensors
    const uint8_t* command_stream,
    size_t command_stream_length,

    const int8_t* bias_and_weights,
    size_t bias_and_weights_length,

    const int8_t* lif_params,
    size_t lif_params_length,

    const int8_t* luts,
    size_t luts_length,


    int is_last_layer,
    size_t out_spk_sum_relative_addr,
    float out_spk_sum_scale,
    int out_spk_sum_zero_point,

    // Non-const tensors
    size_t num_non_const_tensors,
    size_t tensor_arena_size,
    int8_t* tensor_arena,

    int8_t* in_spk,
    int8_t* out_spk,

    size_t bias_relative_addr,
    size_t weight_relative_addr,
    size_t v_mem_relative_addr,
    size_t time_not_updated_relative_addr,
    size_t update_nxt_layer_relative_addr,

    size_t input_layer_size,
    size_t output_layer_size,
    size_t bias_tensor_size,
    size_t weight_tensor_size,


    float in_spk_scale,
    int in_spk_zero_point,

    float v_mem_scale,
    int v_mem_zero_point,
    float time_not_updated_scale,
    int time_not_updated_zero_point,

    float out_spk_scale,
    int out_spk_zero_point
);
NN_Model* MLP_Init();

int MLP_Quantize_Inputs(NN_Model* mlp_model, float* in_spk, float* v_mem, float* time_not_updated);

int MLP_Inference(
    NN_Model* mlp_model,

    int8_t** in_spk_arr,
    size_t in_spk_arr_len,

    int8_t* out_spk
);

int MLP_Inference_test_patterns(
    NN_Model* mlp_model,

    //volatile int8_t*** test_patterns,
    //volatile int8_t* test_targets,
    //volatile int8_t test_patterns[][25][784],
    //volatile int8_t test_targets[],

    //size_t num_samples,

    size_t num_time_steps,
    int make_printouts,

    int8_t* out_spk
);

int MLP_Free(NN_Model* mlp_model);




