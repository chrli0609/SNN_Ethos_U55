#pragma once

#include "../include/nn_data_structure.h"
#include "../model.h"


#include "layers/fc_lif_layer_0.h"
#include "layers/fc_lif_layer_1.h"
#include "layers/fc_lif_layer_2.h"
#include "layers/fc_lif_layer_3.h"
#include "layers/fc_lif_layer_4.h"



#define MLP_INPUT_LAYER_SIZE	FC_LIF_LAYER_0_INPUT_LAYER_SIZE
#define MLP_OUTPUT_LAYER_SIZE	FC_LIF_LAYER_4_OUTPUT_LAYER_SIZE

#define MLP_NUM_LAYERS 5
#define MLP_NUM_TIME_STEPS 25

static int8_t fc_lif_layer_0_in_spk[FC_LIF_LAYER_0_INPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_0_tensor_arena[FC_LIF_LAYER_0_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_0_out_spk[FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_1_tensor_arena[FC_LIF_LAYER_1_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_1_out_spk[FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_2_tensor_arena[FC_LIF_LAYER_2_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_2_out_spk[FC_LIF_LAYER_2_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_3_tensor_arena[FC_LIF_LAYER_3_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_3_out_spk[FC_LIF_LAYER_3_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_4_tensor_arena[FC_LIF_LAYER_4_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_4_out_spk[FC_LIF_LAYER_4_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));


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


NNLayer* Init_fc_lif_layer_2() {

	NNLayer* fc_lif_layer_2 = FC_LIF_Layer_Init(
		Getfc_lif_layer_2CMSPointer(),
		Getfc_lif_layer_2CMSLen(),
		Getfc_lif_layer_2WeightsPointer(),
		Getfc_lif_layer_2WeightsLen(),
		Getfc_lif_layer_2LIFParamPointer(),
		Getfc_lif_layer_2LIFParamLen(),
		Getfc_lif_layer_2LUTPointer(),
		Getfc_lif_layer_2LUTLen(),
		FC_LIF_LAYER_2_IS_LAST_LAYER,
		FC_LIF_LAYER_2_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_2_OUT_SPK_SCALE,
		FC_LIF_LAYER_2_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_2_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_2_TENSOR_ARENA_SIZE,
		fc_lif_layer_2_tensor_arena,
		fc_lif_layer_1_out_spk,
		fc_lif_layer_2_out_spk,
		FC_LIF_LAYER_2_BIAS_ADDR,
		FC_LIF_LAYER_2_WEIGHT_ADDR,
		FC_LIF_LAYER_2_V_MEM_ADDR,
		FC_LIF_LAYER_2_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_2_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_2_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_2_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_2_BIAS_LEN,
		FC_LIF_LAYER_2_WEIGHT_LEN,
		FC_LIF_LAYER_2_IN_SPK_SCALE,
		FC_LIF_LAYER_2_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_2_V_MEM_SCALE,
		FC_LIF_LAYER_2_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_2_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_2_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_2_OUT_SPK_SCALE,
		FC_LIF_LAYER_2_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_2;
}


NNLayer* Init_fc_lif_layer_3() {

	NNLayer* fc_lif_layer_3 = FC_LIF_Layer_Init(
		Getfc_lif_layer_3CMSPointer(),
		Getfc_lif_layer_3CMSLen(),
		Getfc_lif_layer_3WeightsPointer(),
		Getfc_lif_layer_3WeightsLen(),
		Getfc_lif_layer_3LIFParamPointer(),
		Getfc_lif_layer_3LIFParamLen(),
		Getfc_lif_layer_3LUTPointer(),
		Getfc_lif_layer_3LUTLen(),
		FC_LIF_LAYER_3_IS_LAST_LAYER,
		FC_LIF_LAYER_3_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_3_OUT_SPK_SCALE,
		FC_LIF_LAYER_3_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_3_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_3_TENSOR_ARENA_SIZE,
		fc_lif_layer_3_tensor_arena,
		fc_lif_layer_2_out_spk,
		fc_lif_layer_3_out_spk,
		FC_LIF_LAYER_3_BIAS_ADDR,
		FC_LIF_LAYER_3_WEIGHT_ADDR,
		FC_LIF_LAYER_3_V_MEM_ADDR,
		FC_LIF_LAYER_3_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_3_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_3_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_3_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_3_BIAS_LEN,
		FC_LIF_LAYER_3_WEIGHT_LEN,
		FC_LIF_LAYER_3_IN_SPK_SCALE,
		FC_LIF_LAYER_3_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_3_V_MEM_SCALE,
		FC_LIF_LAYER_3_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_3_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_3_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_3_OUT_SPK_SCALE,
		FC_LIF_LAYER_3_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_3;
}


NNLayer* Init_fc_lif_layer_4() {

	NNLayer* fc_lif_layer_4 = FC_LIF_Layer_Init(
		Getfc_lif_layer_4CMSPointer(),
		Getfc_lif_layer_4CMSLen(),
		Getfc_lif_layer_4WeightsPointer(),
		Getfc_lif_layer_4WeightsLen(),
		Getfc_lif_layer_4LIFParamPointer(),
		Getfc_lif_layer_4LIFParamLen(),
		Getfc_lif_layer_4LUTPointer(),
		Getfc_lif_layer_4LUTLen(),
		FC_LIF_LAYER_4_IS_LAST_LAYER,
		FC_LIF_LAYER_4_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_4_OUT_SPK_SCALE,
		FC_LIF_LAYER_4_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_4_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_4_TENSOR_ARENA_SIZE,
		fc_lif_layer_4_tensor_arena,
		fc_lif_layer_3_out_spk,
		fc_lif_layer_4_out_spk,
		FC_LIF_LAYER_4_BIAS_ADDR,
		FC_LIF_LAYER_4_WEIGHT_ADDR,
		FC_LIF_LAYER_4_V_MEM_ADDR,
		FC_LIF_LAYER_4_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_4_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_4_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_4_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_4_BIAS_LEN,
		FC_LIF_LAYER_4_WEIGHT_LEN,
		FC_LIF_LAYER_4_IN_SPK_SCALE,
		FC_LIF_LAYER_4_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_4_V_MEM_SCALE,
		FC_LIF_LAYER_4_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_4_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_4_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_4_OUT_SPK_SCALE,
		FC_LIF_LAYER_4_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_4;
}



NNLayer* (*init_layers_func[MLP_NUM_LAYERS]) (void) = {
	Init_fc_lif_layer_0,
	Init_fc_lif_layer_1,
	Init_fc_lif_layer_2,
	Init_fc_lif_layer_3,
	Init_fc_lif_layer_4,
};