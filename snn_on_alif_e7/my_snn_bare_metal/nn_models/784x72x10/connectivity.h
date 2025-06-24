#pragma once

#include "../include/nn_data_structure.h"
#include "../model.h"


#define MODEL_NAME "784x72x10"


#include "layers/fc_lif_layer_0.h"
#include "layers/fc_lif_layer_1.h"



#define MLP_INPUT_LAYER_SIZE	FC_LIF_LAYER_0_INPUT_LAYER_SIZE
#define MLP_OUTPUT_LAYER_SIZE	FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE

#define MLP_NUM_LAYERS 2
#define MLP_NUM_TIME_STEPS 25

static int8_t fc_lif_layer_0_in_spk[FC_LIF_LAYER_0_INPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_0_tensor_arena[FC_LIF_LAYER_0_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_0_out_spk[FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_1_tensor_arena[FC_LIF_LAYER_1_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_1_out_spk[FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));


NNLayer* Init_fc_lif_layer_0() {

	NNLayer* fc_lif_layer_0 = FC_LIF_Layer_Init(
		Getfc_lif_layer_0CMSPointer(),
		Getfc_lif_layer_0CMSLen(),
		Getfc_lif_layer_0WeightsPointer(),
		Getfc_lif_layer_0WeightsLen(),
		Getfc_lif_layer_0LIFParamPointer(),
		Getfc_lif_layer_0LIFParamLen(),
		Getfc_lif_layer_0LUTPointer(),
		Getfc_lif_layer_0LUTLen(),
		FC_LIF_LAYER_0_IS_LAST_LAYER,
		FC_LIF_LAYER_0_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_0_OUT_SPK_SCALE,
		FC_LIF_LAYER_0_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_0_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_0_TENSOR_ARENA_SIZE,
		fc_lif_layer_0_tensor_arena,
		fc_lif_layer_0_in_spk,
		fc_lif_layer_0_out_spk,
		FC_LIF_LAYER_0_BIAS_ADDR,
		FC_LIF_LAYER_0_WEIGHT_ADDR,
		FC_LIF_LAYER_0_V_MEM_ADDR,
		FC_LIF_LAYER_0_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_0_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_0_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_0_BIAS_LEN,
		FC_LIF_LAYER_0_WEIGHT_LEN,
		FC_LIF_LAYER_0_IN_SPK_SCALE,
		FC_LIF_LAYER_0_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_0_V_MEM_SCALE,
		FC_LIF_LAYER_0_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_0_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_0_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_0_OUT_SPK_SCALE,
		FC_LIF_LAYER_0_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_0;
}


NNLayer* Init_fc_lif_layer_1() {

	NNLayer* fc_lif_layer_1 = FC_LIF_Layer_Init(
		Getfc_lif_layer_1CMSPointer(),
		Getfc_lif_layer_1CMSLen(),
		Getfc_lif_layer_1WeightsPointer(),
		Getfc_lif_layer_1WeightsLen(),
		Getfc_lif_layer_1LIFParamPointer(),
		Getfc_lif_layer_1LIFParamLen(),
		Getfc_lif_layer_1LUTPointer(),
		Getfc_lif_layer_1LUTLen(),
		FC_LIF_LAYER_1_IS_LAST_LAYER,
		FC_LIF_LAYER_1_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_1_OUT_SPK_SCALE,
		FC_LIF_LAYER_1_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_1_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_1_TENSOR_ARENA_SIZE,
		fc_lif_layer_1_tensor_arena,
		fc_lif_layer_0_out_spk,
		fc_lif_layer_1_out_spk,
		FC_LIF_LAYER_1_BIAS_ADDR,
		FC_LIF_LAYER_1_WEIGHT_ADDR,
		FC_LIF_LAYER_1_V_MEM_ADDR,
		FC_LIF_LAYER_1_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_1_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_1_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_1_BIAS_LEN,
		FC_LIF_LAYER_1_WEIGHT_LEN,
		FC_LIF_LAYER_1_IN_SPK_SCALE,
		FC_LIF_LAYER_1_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_1_V_MEM_SCALE,
		FC_LIF_LAYER_1_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_1_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_1_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_1_OUT_SPK_SCALE,
		FC_LIF_LAYER_1_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_1;
}



NNLayer* (*init_layers_func[MLP_NUM_LAYERS]) (void) = {
	Init_fc_lif_layer_0,
	Init_fc_lif_layer_1,
};