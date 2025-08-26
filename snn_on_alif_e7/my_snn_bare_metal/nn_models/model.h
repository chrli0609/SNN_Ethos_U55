#pragma once

#include "../include/nn_data_structure.h"







// Include Test Patterns
//#include "connectivity.h"
#include "test_patterns/pattern_0.h"

//// Set NNLayer Tensor Indices
//#define IN_SPK_TENSOR_IDX               0
//#define BIAS_TENSOR_IDX                 1
//#define WEGIHTS_TENSOR_IDX              2
//#define V_MEM_QUANT_IDX                 3
//#define TIME_NOT_UPDATED_QUANT_IDX      4
//#define UPDATE_NXT_LAYER_IDX            5
//#define OUT_SPK_TENSOR_IDX              6

//// Only exists for last layer
//#define OUT_SPK_SUM_TENSOR_IDX          7



//NNLayer* FC_LIF_Layer_Init(

    //// Const Tensors
    //const uint8_t* command_stream,
    //size_t command_stream_length,

    //int8_t** memory_region_ptrs,
    //size_t* memory_region_sizes,
    //size_t num_regions,

    //int8_t* in_spk,
    //size_t input_layer_size,
    //int8_t* out_spk,
    //size_t output_layer_size,

    //// Non-const tensors

    ////size_t bias_relative_addr,
    ////size_t weight_relative_addr,
    ////size_t v_mem_relative_addr,
    ////size_t time_not_updated_relative_addr,
    ////size_t update_nxt_layer_relative_addr,

    ////size_t v_mem_size,
    ////size_t time_not_updated_size,
    ////size_t update_nxt_layer_size,
    ////size_t bias_tensor_size,
    ////size_t weight_tensor_size,

    //char** tensor_names,
    //size_t* tensor_relative_addrs,
    //size_t* tensor_regions,
    //size_t* tensor_sizes,
    //size_t num_tensors,


    ////float in_spk_scale,
    ////int in_spk_zero_point,

    ////float v_mem_scale,
    ////int v_mem_zero_point,
    ////float time_not_updated_scale,
    ////int time_not_updated_zero_point,

    ////float out_spk_scale,
    ////int out_spk_zero_point
    //int is_last_layer
//);

NN_Model* MLP_Init();

int MLP_Quantize_Inputs(NN_Model* mlp_model, float* in_spk, float* v_mem, float* time_not_updated);

//int MLP_Inference(
    //NN_Model* mlp_model,

    //int8_t** in_spk_arr,
    //size_t in_spk_arr_len,

    //int8_t* out_spk
//);

int MLP_Inference_test_patterns(
    NN_Model* mlp_model,

    //volatile int8_t* test_patterns,
    //volatile int8_t* test_targets,
    //volatile int8_t test_patterns[test_input_0_NUM_SAMPLES][2][16],
    volatile int8_t test_patterns[test_input_0_NUM_SAMPLES][25][784],
    //volatile int8_t test_patterns[test_input_0_NUM_SAMPLES][MLP_NUM_TIME_STEPS][MLP_INPUT_LAYER_SIZE],
    volatile int8_t test_targets[test_input_0_NUM_SAMPLES],

    size_t num_samples,

    int make_printouts
);

int MLP_Free(NN_Model* mlp_model);





/*
Set region for every layer

*/

//static int8_t fc_lif_layer_0_tensor_arena[FC_LIF_LAYER_0_TENSOR_ARENA_SIZE] __attribute__((section("model_params_dtcm"))) __attribute__((aligned(16)));
//static int8_t fc_lif_layer_0_in_spk[FC_LIF_LAYER_0_INPUT_LAYER_SIZE] __attribute__((section("model_params_dtcm"))) __attribute__((aligned(16)));
//static int8_t fc_lif_layer_0_out_spk[FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_dtcm"))) __attribute__((aligned(16)));




//NN_Model_CPU* Init_CPU_MLP();

//NNLayer_CPU* FC_LIF_Layer_CPU_Init(

    //// Const tensors
    //float* weights_arr,
    //float* biases_arr,
    //float* beta_arr,
    //float* vth_arr,

    //// Non-const tensors
    //float* in_spk,
    //float* out_spk,
    //float* v_mem,
    //float* time_since_last_update,

    //size_t input_size,
    //size_t output_size,

    //float* out_spk_sum
//);

//int MLP_Inference_CPU_test_patterns(
    //NN_Model_CPU* mlp_model,

    //volatile int8_t test_patterns[test_input_0_NUM_SAMPLES][25][784],
    //volatile int8_t test_targets[test_input_0_NUM_SAMPLES],

    //size_t num_samples,
    //size_t num_time_steps,

    //int make_printouts
//);