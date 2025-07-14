#pragma once

#include "../include/nn_data_structure.h"
#include "../model.h"


#define MODEL_NAME "784x48^14x10"


#include "layers/fc_lif_layer_0.h"
#include "layers/fc_lif_layer_1.h"
#include "layers/fc_lif_layer_2.h"
#include "layers/fc_lif_layer_3.h"
#include "layers/fc_lif_layer_4.h"
#include "layers/fc_lif_layer_5.h"
#include "layers/fc_lif_layer_6.h"
#include "layers/fc_lif_layer_7.h"
#include "layers/fc_lif_layer_8.h"
#include "layers/fc_lif_layer_9.h"
#include "layers/fc_lif_layer_10.h"
#include "layers/fc_lif_layer_11.h"
#include "layers/fc_lif_layer_12.h"
#include "layers/fc_lif_layer_13.h"
#include "layers/fc_lif_layer_14.h"



#define MLP_INPUT_LAYER_SIZE	FC_LIF_LAYER_0_INPUT_LAYER_SIZE
#define MLP_OUTPUT_LAYER_SIZE	FC_LIF_LAYER_14_OUTPUT_LAYER_SIZE

#define MLP_NUM_LAYERS 15
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

static int8_t fc_lif_layer_5_tensor_arena[FC_LIF_LAYER_5_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_5_out_spk[FC_LIF_LAYER_5_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_6_tensor_arena[FC_LIF_LAYER_6_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_6_out_spk[FC_LIF_LAYER_6_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_7_tensor_arena[FC_LIF_LAYER_7_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_7_out_spk[FC_LIF_LAYER_7_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_8_tensor_arena[FC_LIF_LAYER_8_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_8_out_spk[FC_LIF_LAYER_8_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_9_tensor_arena[FC_LIF_LAYER_9_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_9_out_spk[FC_LIF_LAYER_9_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_10_tensor_arena[FC_LIF_LAYER_10_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_10_out_spk[FC_LIF_LAYER_10_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_11_tensor_arena[FC_LIF_LAYER_11_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_11_out_spk[FC_LIF_LAYER_11_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_12_tensor_arena[FC_LIF_LAYER_12_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_12_out_spk[FC_LIF_LAYER_12_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_13_tensor_arena[FC_LIF_LAYER_13_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_13_out_spk[FC_LIF_LAYER_13_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_14_tensor_arena[FC_LIF_LAYER_14_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_14_out_spk[FC_LIF_LAYER_14_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));


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


NNLayer* Init_fc_lif_layer_5() {

	NNLayer* fc_lif_layer_5 = FC_LIF_Layer_Init(
		Getfc_lif_layer_5CMSPointer(),
		Getfc_lif_layer_5CMSLen(),
		Getfc_lif_layer_5WeightsPointer(),
		Getfc_lif_layer_5WeightsLen(),
		Getfc_lif_layer_5LIFParamPointer(),
		Getfc_lif_layer_5LIFParamLen(),
		Getfc_lif_layer_5LUTPointer(),
		Getfc_lif_layer_5LUTLen(),
		FC_LIF_LAYER_5_IS_LAST_LAYER,
		FC_LIF_LAYER_5_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_5_OUT_SPK_SCALE,
		FC_LIF_LAYER_5_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_5_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_5_TENSOR_ARENA_SIZE,
		fc_lif_layer_5_tensor_arena,
		fc_lif_layer_4_out_spk,
		fc_lif_layer_5_out_spk,
		FC_LIF_LAYER_5_BIAS_ADDR,
		FC_LIF_LAYER_5_WEIGHT_ADDR,
		FC_LIF_LAYER_5_V_MEM_ADDR,
		FC_LIF_LAYER_5_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_5_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_5_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_5_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_5_BIAS_LEN,
		FC_LIF_LAYER_5_WEIGHT_LEN,
		FC_LIF_LAYER_5_IN_SPK_SCALE,
		FC_LIF_LAYER_5_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_5_V_MEM_SCALE,
		FC_LIF_LAYER_5_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_5_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_5_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_5_OUT_SPK_SCALE,
		FC_LIF_LAYER_5_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_5;
}


NNLayer* Init_fc_lif_layer_6() {

	NNLayer* fc_lif_layer_6 = FC_LIF_Layer_Init(
		Getfc_lif_layer_6CMSPointer(),
		Getfc_lif_layer_6CMSLen(),
		Getfc_lif_layer_6WeightsPointer(),
		Getfc_lif_layer_6WeightsLen(),
		Getfc_lif_layer_6LIFParamPointer(),
		Getfc_lif_layer_6LIFParamLen(),
		Getfc_lif_layer_6LUTPointer(),
		Getfc_lif_layer_6LUTLen(),
		FC_LIF_LAYER_6_IS_LAST_LAYER,
		FC_LIF_LAYER_6_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_6_OUT_SPK_SCALE,
		FC_LIF_LAYER_6_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_6_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_6_TENSOR_ARENA_SIZE,
		fc_lif_layer_6_tensor_arena,
		fc_lif_layer_5_out_spk,
		fc_lif_layer_6_out_spk,
		FC_LIF_LAYER_6_BIAS_ADDR,
		FC_LIF_LAYER_6_WEIGHT_ADDR,
		FC_LIF_LAYER_6_V_MEM_ADDR,
		FC_LIF_LAYER_6_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_6_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_6_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_6_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_6_BIAS_LEN,
		FC_LIF_LAYER_6_WEIGHT_LEN,
		FC_LIF_LAYER_6_IN_SPK_SCALE,
		FC_LIF_LAYER_6_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_6_V_MEM_SCALE,
		FC_LIF_LAYER_6_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_6_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_6_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_6_OUT_SPK_SCALE,
		FC_LIF_LAYER_6_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_6;
}


NNLayer* Init_fc_lif_layer_7() {

	NNLayer* fc_lif_layer_7 = FC_LIF_Layer_Init(
		Getfc_lif_layer_7CMSPointer(),
		Getfc_lif_layer_7CMSLen(),
		Getfc_lif_layer_7WeightsPointer(),
		Getfc_lif_layer_7WeightsLen(),
		Getfc_lif_layer_7LIFParamPointer(),
		Getfc_lif_layer_7LIFParamLen(),
		Getfc_lif_layer_7LUTPointer(),
		Getfc_lif_layer_7LUTLen(),
		FC_LIF_LAYER_7_IS_LAST_LAYER,
		FC_LIF_LAYER_7_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_7_OUT_SPK_SCALE,
		FC_LIF_LAYER_7_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_7_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_7_TENSOR_ARENA_SIZE,
		fc_lif_layer_7_tensor_arena,
		fc_lif_layer_6_out_spk,
		fc_lif_layer_7_out_spk,
		FC_LIF_LAYER_7_BIAS_ADDR,
		FC_LIF_LAYER_7_WEIGHT_ADDR,
		FC_LIF_LAYER_7_V_MEM_ADDR,
		FC_LIF_LAYER_7_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_7_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_7_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_7_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_7_BIAS_LEN,
		FC_LIF_LAYER_7_WEIGHT_LEN,
		FC_LIF_LAYER_7_IN_SPK_SCALE,
		FC_LIF_LAYER_7_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_7_V_MEM_SCALE,
		FC_LIF_LAYER_7_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_7_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_7_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_7_OUT_SPK_SCALE,
		FC_LIF_LAYER_7_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_7;
}


NNLayer* Init_fc_lif_layer_8() {

	NNLayer* fc_lif_layer_8 = FC_LIF_Layer_Init(
		Getfc_lif_layer_8CMSPointer(),
		Getfc_lif_layer_8CMSLen(),
		Getfc_lif_layer_8WeightsPointer(),
		Getfc_lif_layer_8WeightsLen(),
		Getfc_lif_layer_8LIFParamPointer(),
		Getfc_lif_layer_8LIFParamLen(),
		Getfc_lif_layer_8LUTPointer(),
		Getfc_lif_layer_8LUTLen(),
		FC_LIF_LAYER_8_IS_LAST_LAYER,
		FC_LIF_LAYER_8_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_8_OUT_SPK_SCALE,
		FC_LIF_LAYER_8_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_8_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_8_TENSOR_ARENA_SIZE,
		fc_lif_layer_8_tensor_arena,
		fc_lif_layer_7_out_spk,
		fc_lif_layer_8_out_spk,
		FC_LIF_LAYER_8_BIAS_ADDR,
		FC_LIF_LAYER_8_WEIGHT_ADDR,
		FC_LIF_LAYER_8_V_MEM_ADDR,
		FC_LIF_LAYER_8_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_8_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_8_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_8_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_8_BIAS_LEN,
		FC_LIF_LAYER_8_WEIGHT_LEN,
		FC_LIF_LAYER_8_IN_SPK_SCALE,
		FC_LIF_LAYER_8_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_8_V_MEM_SCALE,
		FC_LIF_LAYER_8_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_8_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_8_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_8_OUT_SPK_SCALE,
		FC_LIF_LAYER_8_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_8;
}


NNLayer* Init_fc_lif_layer_9() {

	NNLayer* fc_lif_layer_9 = FC_LIF_Layer_Init(
		Getfc_lif_layer_9CMSPointer(),
		Getfc_lif_layer_9CMSLen(),
		Getfc_lif_layer_9WeightsPointer(),
		Getfc_lif_layer_9WeightsLen(),
		Getfc_lif_layer_9LIFParamPointer(),
		Getfc_lif_layer_9LIFParamLen(),
		Getfc_lif_layer_9LUTPointer(),
		Getfc_lif_layer_9LUTLen(),
		FC_LIF_LAYER_9_IS_LAST_LAYER,
		FC_LIF_LAYER_9_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_9_OUT_SPK_SCALE,
		FC_LIF_LAYER_9_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_9_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_9_TENSOR_ARENA_SIZE,
		fc_lif_layer_9_tensor_arena,
		fc_lif_layer_8_out_spk,
		fc_lif_layer_9_out_spk,
		FC_LIF_LAYER_9_BIAS_ADDR,
		FC_LIF_LAYER_9_WEIGHT_ADDR,
		FC_LIF_LAYER_9_V_MEM_ADDR,
		FC_LIF_LAYER_9_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_9_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_9_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_9_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_9_BIAS_LEN,
		FC_LIF_LAYER_9_WEIGHT_LEN,
		FC_LIF_LAYER_9_IN_SPK_SCALE,
		FC_LIF_LAYER_9_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_9_V_MEM_SCALE,
		FC_LIF_LAYER_9_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_9_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_9_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_9_OUT_SPK_SCALE,
		FC_LIF_LAYER_9_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_9;
}


NNLayer* Init_fc_lif_layer_10() {

	NNLayer* fc_lif_layer_10 = FC_LIF_Layer_Init(
		Getfc_lif_layer_10CMSPointer(),
		Getfc_lif_layer_10CMSLen(),
		Getfc_lif_layer_10WeightsPointer(),
		Getfc_lif_layer_10WeightsLen(),
		Getfc_lif_layer_10LIFParamPointer(),
		Getfc_lif_layer_10LIFParamLen(),
		Getfc_lif_layer_10LUTPointer(),
		Getfc_lif_layer_10LUTLen(),
		FC_LIF_LAYER_10_IS_LAST_LAYER,
		FC_LIF_LAYER_10_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_10_OUT_SPK_SCALE,
		FC_LIF_LAYER_10_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_10_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_10_TENSOR_ARENA_SIZE,
		fc_lif_layer_10_tensor_arena,
		fc_lif_layer_9_out_spk,
		fc_lif_layer_10_out_spk,
		FC_LIF_LAYER_10_BIAS_ADDR,
		FC_LIF_LAYER_10_WEIGHT_ADDR,
		FC_LIF_LAYER_10_V_MEM_ADDR,
		FC_LIF_LAYER_10_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_10_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_10_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_10_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_10_BIAS_LEN,
		FC_LIF_LAYER_10_WEIGHT_LEN,
		FC_LIF_LAYER_10_IN_SPK_SCALE,
		FC_LIF_LAYER_10_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_10_V_MEM_SCALE,
		FC_LIF_LAYER_10_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_10_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_10_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_10_OUT_SPK_SCALE,
		FC_LIF_LAYER_10_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_10;
}


NNLayer* Init_fc_lif_layer_11() {

	NNLayer* fc_lif_layer_11 = FC_LIF_Layer_Init(
		Getfc_lif_layer_11CMSPointer(),
		Getfc_lif_layer_11CMSLen(),
		Getfc_lif_layer_11WeightsPointer(),
		Getfc_lif_layer_11WeightsLen(),
		Getfc_lif_layer_11LIFParamPointer(),
		Getfc_lif_layer_11LIFParamLen(),
		Getfc_lif_layer_11LUTPointer(),
		Getfc_lif_layer_11LUTLen(),
		FC_LIF_LAYER_11_IS_LAST_LAYER,
		FC_LIF_LAYER_11_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_11_OUT_SPK_SCALE,
		FC_LIF_LAYER_11_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_11_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_11_TENSOR_ARENA_SIZE,
		fc_lif_layer_11_tensor_arena,
		fc_lif_layer_10_out_spk,
		fc_lif_layer_11_out_spk,
		FC_LIF_LAYER_11_BIAS_ADDR,
		FC_LIF_LAYER_11_WEIGHT_ADDR,
		FC_LIF_LAYER_11_V_MEM_ADDR,
		FC_LIF_LAYER_11_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_11_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_11_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_11_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_11_BIAS_LEN,
		FC_LIF_LAYER_11_WEIGHT_LEN,
		FC_LIF_LAYER_11_IN_SPK_SCALE,
		FC_LIF_LAYER_11_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_11_V_MEM_SCALE,
		FC_LIF_LAYER_11_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_11_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_11_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_11_OUT_SPK_SCALE,
		FC_LIF_LAYER_11_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_11;
}


NNLayer* Init_fc_lif_layer_12() {

	NNLayer* fc_lif_layer_12 = FC_LIF_Layer_Init(
		Getfc_lif_layer_12CMSPointer(),
		Getfc_lif_layer_12CMSLen(),
		Getfc_lif_layer_12WeightsPointer(),
		Getfc_lif_layer_12WeightsLen(),
		Getfc_lif_layer_12LIFParamPointer(),
		Getfc_lif_layer_12LIFParamLen(),
		Getfc_lif_layer_12LUTPointer(),
		Getfc_lif_layer_12LUTLen(),
		FC_LIF_LAYER_12_IS_LAST_LAYER,
		FC_LIF_LAYER_12_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_12_OUT_SPK_SCALE,
		FC_LIF_LAYER_12_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_12_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_12_TENSOR_ARENA_SIZE,
		fc_lif_layer_12_tensor_arena,
		fc_lif_layer_11_out_spk,
		fc_lif_layer_12_out_spk,
		FC_LIF_LAYER_12_BIAS_ADDR,
		FC_LIF_LAYER_12_WEIGHT_ADDR,
		FC_LIF_LAYER_12_V_MEM_ADDR,
		FC_LIF_LAYER_12_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_12_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_12_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_12_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_12_BIAS_LEN,
		FC_LIF_LAYER_12_WEIGHT_LEN,
		FC_LIF_LAYER_12_IN_SPK_SCALE,
		FC_LIF_LAYER_12_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_12_V_MEM_SCALE,
		FC_LIF_LAYER_12_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_12_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_12_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_12_OUT_SPK_SCALE,
		FC_LIF_LAYER_12_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_12;
}


NNLayer* Init_fc_lif_layer_13() {

	NNLayer* fc_lif_layer_13 = FC_LIF_Layer_Init(
		Getfc_lif_layer_13CMSPointer(),
		Getfc_lif_layer_13CMSLen(),
		Getfc_lif_layer_13WeightsPointer(),
		Getfc_lif_layer_13WeightsLen(),
		Getfc_lif_layer_13LIFParamPointer(),
		Getfc_lif_layer_13LIFParamLen(),
		Getfc_lif_layer_13LUTPointer(),
		Getfc_lif_layer_13LUTLen(),
		FC_LIF_LAYER_13_IS_LAST_LAYER,
		FC_LIF_LAYER_13_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_13_OUT_SPK_SCALE,
		FC_LIF_LAYER_13_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_13_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_13_TENSOR_ARENA_SIZE,
		fc_lif_layer_13_tensor_arena,
		fc_lif_layer_12_out_spk,
		fc_lif_layer_13_out_spk,
		FC_LIF_LAYER_13_BIAS_ADDR,
		FC_LIF_LAYER_13_WEIGHT_ADDR,
		FC_LIF_LAYER_13_V_MEM_ADDR,
		FC_LIF_LAYER_13_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_13_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_13_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_13_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_13_BIAS_LEN,
		FC_LIF_LAYER_13_WEIGHT_LEN,
		FC_LIF_LAYER_13_IN_SPK_SCALE,
		FC_LIF_LAYER_13_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_13_V_MEM_SCALE,
		FC_LIF_LAYER_13_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_13_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_13_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_13_OUT_SPK_SCALE,
		FC_LIF_LAYER_13_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_13;
}


NNLayer* Init_fc_lif_layer_14() {

	NNLayer* fc_lif_layer_14 = FC_LIF_Layer_Init(
		Getfc_lif_layer_14CMSPointer(),
		Getfc_lif_layer_14CMSLen(),
		Getfc_lif_layer_14WeightsPointer(),
		Getfc_lif_layer_14WeightsLen(),
		Getfc_lif_layer_14LIFParamPointer(),
		Getfc_lif_layer_14LIFParamLen(),
		Getfc_lif_layer_14LUTPointer(),
		Getfc_lif_layer_14LUTLen(),
		FC_LIF_LAYER_14_IS_LAST_LAYER,
		FC_LIF_LAYER_14_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_14_OUT_SPK_SCALE,
		FC_LIF_LAYER_14_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_14_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_14_TENSOR_ARENA_SIZE,
		fc_lif_layer_14_tensor_arena,
		fc_lif_layer_13_out_spk,
		fc_lif_layer_14_out_spk,
		FC_LIF_LAYER_14_BIAS_ADDR,
		FC_LIF_LAYER_14_WEIGHT_ADDR,
		FC_LIF_LAYER_14_V_MEM_ADDR,
		FC_LIF_LAYER_14_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_14_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_14_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_14_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_14_BIAS_LEN,
		FC_LIF_LAYER_14_WEIGHT_LEN,
		FC_LIF_LAYER_14_IN_SPK_SCALE,
		FC_LIF_LAYER_14_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_14_V_MEM_SCALE,
		FC_LIF_LAYER_14_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_14_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_14_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_14_OUT_SPK_SCALE,
		FC_LIF_LAYER_14_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_14;
}



NNLayer* (*init_layers_func[MLP_NUM_LAYERS]) (void) = {
	Init_fc_lif_layer_0,
	Init_fc_lif_layer_1,
	Init_fc_lif_layer_2,
	Init_fc_lif_layer_3,
	Init_fc_lif_layer_4,
	Init_fc_lif_layer_5,
	Init_fc_lif_layer_6,
	Init_fc_lif_layer_7,
	Init_fc_lif_layer_8,
	Init_fc_lif_layer_9,
	Init_fc_lif_layer_10,
	Init_fc_lif_layer_11,
	Init_fc_lif_layer_12,
	Init_fc_lif_layer_13,
	Init_fc_lif_layer_14,
};