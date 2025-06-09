#pragma once

#include "include/nn_data_structure.h"




// Include Layers
#include "nn_models/nmnist_784x64x64x10/layers/fc_lif_layer_0.h"
#include "nn_models/nmnist_784x64x64x10/layers/fc_lif_layer_1.h"
#include "nn_models/nmnist_784x64x64x10/layers/fc_lif_layer_2.h"

/// Include Test Patterns
#include "nn_models/nmnist_784x64x64x10/test_patterns/pattern_0.h"

// Set model input and output sizes
#define MLP_INPUT_LAYER_SIZE    FC_LIF_LAYER_0_INPUT_LAYER_SIZE
#define MLP_OUTPUT_LAYER_SIZE   FC_LIF_LAYER_2_OUTPUT_LAYER_SIZE



#define MLP_NUM_TIME_STEPS 25


// Set NNLayer Tensor Indices
#define IN_SPK_TENSOR_IDX               0
#define BIAS_TENSOR_IDX                 1
#define WEGIHTS_TENSOR_IDX              2
#define V_MEM_QUANT_IDX                 3
#define TIME_NOT_UPDATED_QUANT_IDX      4
#define UPDATE_NXT_LAYER_IDX            5
#define OUT_SPK_TENSOR_IDX              6




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

    //int8_t*** test_patterns,
    //int8_t* test_targets,
    volatile int8_t test_patterns[test_input_0_NUM_SAMPLES][MLP_NUM_TIME_STEPS][MLP_INPUT_LAYER_SIZE],
    volatile int8_t test_targets[test_input_0_NUM_SAMPLES],

    size_t num_samples,

    size_t num_time_steps,

    int8_t* out_spk
);

int MLP_Free(NN_Model* mlp_model);






/*
Set region for every layer

*/

//static int8_t nnlayer0_tensor_arena[FC_LIF_LAYER_0_TENSOR_ARENA_SIZE] __attribute__((section("model_params_dtcm"))) __attribute__((aligned(16)));
//static int8_t nnlayer0_in_spk[FC_LIF_LAYER_0_INPUT_LAYER_SIZE] __attribute__((section("model_params_dtcm"))) __attribute__((aligned(16)));
//static int8_t nnlayer0_out_spk[FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_dtcm"))) __attribute__((aligned(16)));
static int8_t nnlayer0_tensor_arena[FC_LIF_LAYER_0_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t nnlayer0_in_spk[FC_LIF_LAYER_0_INPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t nnlayer0_out_spk[FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t nnlayer1_tensor_arena[FC_LIF_LAYER_1_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t nnlayer1_out_spk[FC_LIF_LAYER_1_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));


static int8_t nnlayer2_tensor_arena[FC_LIF_LAYER_1_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t nnlayer2_out_spk[FC_LIF_LAYER_1_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
