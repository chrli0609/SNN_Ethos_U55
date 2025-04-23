#pragma once


#include "include/nn_data_structure.h"



NN_Model* MLP_Init();

int MLP_Quantize_Inputs(NN_Model* mlp_model, float* in_spk, float* ln_beta, float* vth, float* v_mem, float* time_not_updated);

int MLP_Inference(
    NN_Model* mlp_model,

    float* in_spk,

    float* ln_beta,
    float* vth,
    float* v_mem,
    float* time_not_updated,

    float* out_spk
);

int MLP_Free(NN_Model* mlp_model);