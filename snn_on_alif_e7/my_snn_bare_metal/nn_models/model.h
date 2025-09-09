#pragma once

#include "../include/nn_data_structure.h"






NN_Model* MLP_Init();

int MLP_Quantize_Inputs(NN_Model* mlp_model, float* in_spk, float* v_mem, float* time_not_updated);


int MLP_Inference_test_patterns(
    NN_Model* mlp_model,
    int make_printouts
);

int MLP_Free(NN_Model* mlp_model);

