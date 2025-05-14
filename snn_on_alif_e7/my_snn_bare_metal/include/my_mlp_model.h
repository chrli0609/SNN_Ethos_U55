#pragma once


#include "include/nn_data_structure.h"


// First Layer
#include "include/fc_lif_layer_0.h"

#include "include/fc_lif_layer_1.h"


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