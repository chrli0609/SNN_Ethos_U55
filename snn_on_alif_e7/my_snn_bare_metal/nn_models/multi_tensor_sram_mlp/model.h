#pragma once

#include "include/nn_data_structure.h"


// Include Layers
#include "nn_models/multi_tensor_sram_mlp/layers/fc_lif_layer_0.h"
#include "nn_models/multi_tensor_sram_mlp/layers/fc_lif_layer_1.h"


// Set model input and output sizes
#define MLP_INPUT_LAYER_SIZE    FC_LIF_LAYER_0_INPUT_LAYER_SIZE
#define MLP_OUTPUT_LAYER_SIZE   FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE


NN_Model* MLP_Init();

int MLP_Quantize_Inputs(NN_Model* mlp_model, float* in_spk, float* v_mem, float* time_not_updated);

int MLP_Inference(
    NN_Model* mlp_model,

    int8_t** in_spk_arr,
    size_t in_spk_arr_len,

    int8_t* out_spk
);

int MLP_Free(NN_Model* mlp_model);






/*
Set region for every layer

*/

static int8_t nnlayer0_tensor_arena[FC_LIF_LAYER_0_TENSOR_ARENA_SIZE] __attribute__((section("model_params_dtcm"))) __attribute__((aligned(16)));
static int8_t nnlayer1_tensor_arena[FC_LIF_LAYER_1_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram0"))) __attribute__((aligned(16)));
