#pragma once

#include "../include/nn_data_structure.h"
#include "../model.h"


#define MODEL_NAME "784x48^50x10"


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
#include "layers/fc_lif_layer_15.h"
#include "layers/fc_lif_layer_16.h"
#include "layers/fc_lif_layer_17.h"
#include "layers/fc_lif_layer_18.h"
#include "layers/fc_lif_layer_19.h"
#include "layers/fc_lif_layer_20.h"
#include "layers/fc_lif_layer_21.h"
#include "layers/fc_lif_layer_22.h"
#include "layers/fc_lif_layer_23.h"
#include "layers/fc_lif_layer_24.h"
#include "layers/fc_lif_layer_25.h"
#include "layers/fc_lif_layer_26.h"
#include "layers/fc_lif_layer_27.h"
#include "layers/fc_lif_layer_28.h"
#include "layers/fc_lif_layer_29.h"
#include "layers/fc_lif_layer_30.h"
#include "layers/fc_lif_layer_31.h"
#include "layers/fc_lif_layer_32.h"
#include "layers/fc_lif_layer_33.h"
#include "layers/fc_lif_layer_34.h"
#include "layers/fc_lif_layer_35.h"
#include "layers/fc_lif_layer_36.h"
#include "layers/fc_lif_layer_37.h"
#include "layers/fc_lif_layer_38.h"
#include "layers/fc_lif_layer_39.h"
#include "layers/fc_lif_layer_40.h"
#include "layers/fc_lif_layer_41.h"
#include "layers/fc_lif_layer_42.h"
#include "layers/fc_lif_layer_43.h"
#include "layers/fc_lif_layer_44.h"
#include "layers/fc_lif_layer_45.h"
#include "layers/fc_lif_layer_46.h"
#include "layers/fc_lif_layer_47.h"
#include "layers/fc_lif_layer_48.h"
#include "layers/fc_lif_layer_49.h"
#include "layers/fc_lif_layer_50.h"



#define MLP_INPUT_LAYER_SIZE	FC_LIF_LAYER_0_INPUT_LAYER_SIZE
#define MLP_OUTPUT_LAYER_SIZE	FC_LIF_LAYER_50_OUTPUT_LAYER_SIZE

#define MLP_NUM_LAYERS 51
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

static int8_t fc_lif_layer_15_tensor_arena[FC_LIF_LAYER_15_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_15_out_spk[FC_LIF_LAYER_15_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_16_tensor_arena[FC_LIF_LAYER_16_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_16_out_spk[FC_LIF_LAYER_16_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_17_tensor_arena[FC_LIF_LAYER_17_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_17_out_spk[FC_LIF_LAYER_17_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_18_tensor_arena[FC_LIF_LAYER_18_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_18_out_spk[FC_LIF_LAYER_18_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_19_tensor_arena[FC_LIF_LAYER_19_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_19_out_spk[FC_LIF_LAYER_19_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_20_tensor_arena[FC_LIF_LAYER_20_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_20_out_spk[FC_LIF_LAYER_20_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_21_tensor_arena[FC_LIF_LAYER_21_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_21_out_spk[FC_LIF_LAYER_21_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_22_tensor_arena[FC_LIF_LAYER_22_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_22_out_spk[FC_LIF_LAYER_22_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_23_tensor_arena[FC_LIF_LAYER_23_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_23_out_spk[FC_LIF_LAYER_23_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_24_tensor_arena[FC_LIF_LAYER_24_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_24_out_spk[FC_LIF_LAYER_24_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_25_tensor_arena[FC_LIF_LAYER_25_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_25_out_spk[FC_LIF_LAYER_25_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_26_tensor_arena[FC_LIF_LAYER_26_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_26_out_spk[FC_LIF_LAYER_26_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_27_tensor_arena[FC_LIF_LAYER_27_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_27_out_spk[FC_LIF_LAYER_27_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_28_tensor_arena[FC_LIF_LAYER_28_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_28_out_spk[FC_LIF_LAYER_28_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_29_tensor_arena[FC_LIF_LAYER_29_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_29_out_spk[FC_LIF_LAYER_29_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_30_tensor_arena[FC_LIF_LAYER_30_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_30_out_spk[FC_LIF_LAYER_30_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_31_tensor_arena[FC_LIF_LAYER_31_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_31_out_spk[FC_LIF_LAYER_31_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_32_tensor_arena[FC_LIF_LAYER_32_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_32_out_spk[FC_LIF_LAYER_32_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_33_tensor_arena[FC_LIF_LAYER_33_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_33_out_spk[FC_LIF_LAYER_33_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_34_tensor_arena[FC_LIF_LAYER_34_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_34_out_spk[FC_LIF_LAYER_34_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_35_tensor_arena[FC_LIF_LAYER_35_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_35_out_spk[FC_LIF_LAYER_35_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_36_tensor_arena[FC_LIF_LAYER_36_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_36_out_spk[FC_LIF_LAYER_36_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_37_tensor_arena[FC_LIF_LAYER_37_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_37_out_spk[FC_LIF_LAYER_37_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_38_tensor_arena[FC_LIF_LAYER_38_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_38_out_spk[FC_LIF_LAYER_38_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_39_tensor_arena[FC_LIF_LAYER_39_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_39_out_spk[FC_LIF_LAYER_39_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_40_tensor_arena[FC_LIF_LAYER_40_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_40_out_spk[FC_LIF_LAYER_40_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_41_tensor_arena[FC_LIF_LAYER_41_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_41_out_spk[FC_LIF_LAYER_41_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_42_tensor_arena[FC_LIF_LAYER_42_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_42_out_spk[FC_LIF_LAYER_42_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_43_tensor_arena[FC_LIF_LAYER_43_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_43_out_spk[FC_LIF_LAYER_43_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_44_tensor_arena[FC_LIF_LAYER_44_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_44_out_spk[FC_LIF_LAYER_44_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_45_tensor_arena[FC_LIF_LAYER_45_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_45_out_spk[FC_LIF_LAYER_45_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_46_tensor_arena[FC_LIF_LAYER_46_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_46_out_spk[FC_LIF_LAYER_46_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_47_tensor_arena[FC_LIF_LAYER_47_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_47_out_spk[FC_LIF_LAYER_47_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_48_tensor_arena[FC_LIF_LAYER_48_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_48_out_spk[FC_LIF_LAYER_48_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_49_tensor_arena[FC_LIF_LAYER_49_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_49_out_spk[FC_LIF_LAYER_49_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_50_tensor_arena[FC_LIF_LAYER_50_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_50_out_spk[FC_LIF_LAYER_50_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));


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


NNLayer* Init_fc_lif_layer_15() {

	NNLayer* fc_lif_layer_15 = FC_LIF_Layer_Init(
		Getfc_lif_layer_15CMSPointer(),
		Getfc_lif_layer_15CMSLen(),
		Getfc_lif_layer_15WeightsPointer(),
		Getfc_lif_layer_15WeightsLen(),
		Getfc_lif_layer_15LIFParamPointer(),
		Getfc_lif_layer_15LIFParamLen(),
		Getfc_lif_layer_15LUTPointer(),
		Getfc_lif_layer_15LUTLen(),
		FC_LIF_LAYER_15_IS_LAST_LAYER,
		FC_LIF_LAYER_15_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_15_OUT_SPK_SCALE,
		FC_LIF_LAYER_15_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_15_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_15_TENSOR_ARENA_SIZE,
		fc_lif_layer_15_tensor_arena,
		fc_lif_layer_14_out_spk,
		fc_lif_layer_15_out_spk,
		FC_LIF_LAYER_15_BIAS_ADDR,
		FC_LIF_LAYER_15_WEIGHT_ADDR,
		FC_LIF_LAYER_15_V_MEM_ADDR,
		FC_LIF_LAYER_15_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_15_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_15_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_15_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_15_BIAS_LEN,
		FC_LIF_LAYER_15_WEIGHT_LEN,
		FC_LIF_LAYER_15_IN_SPK_SCALE,
		FC_LIF_LAYER_15_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_15_V_MEM_SCALE,
		FC_LIF_LAYER_15_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_15_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_15_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_15_OUT_SPK_SCALE,
		FC_LIF_LAYER_15_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_15;
}


NNLayer* Init_fc_lif_layer_16() {

	NNLayer* fc_lif_layer_16 = FC_LIF_Layer_Init(
		Getfc_lif_layer_16CMSPointer(),
		Getfc_lif_layer_16CMSLen(),
		Getfc_lif_layer_16WeightsPointer(),
		Getfc_lif_layer_16WeightsLen(),
		Getfc_lif_layer_16LIFParamPointer(),
		Getfc_lif_layer_16LIFParamLen(),
		Getfc_lif_layer_16LUTPointer(),
		Getfc_lif_layer_16LUTLen(),
		FC_LIF_LAYER_16_IS_LAST_LAYER,
		FC_LIF_LAYER_16_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_16_OUT_SPK_SCALE,
		FC_LIF_LAYER_16_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_16_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_16_TENSOR_ARENA_SIZE,
		fc_lif_layer_16_tensor_arena,
		fc_lif_layer_15_out_spk,
		fc_lif_layer_16_out_spk,
		FC_LIF_LAYER_16_BIAS_ADDR,
		FC_LIF_LAYER_16_WEIGHT_ADDR,
		FC_LIF_LAYER_16_V_MEM_ADDR,
		FC_LIF_LAYER_16_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_16_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_16_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_16_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_16_BIAS_LEN,
		FC_LIF_LAYER_16_WEIGHT_LEN,
		FC_LIF_LAYER_16_IN_SPK_SCALE,
		FC_LIF_LAYER_16_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_16_V_MEM_SCALE,
		FC_LIF_LAYER_16_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_16_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_16_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_16_OUT_SPK_SCALE,
		FC_LIF_LAYER_16_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_16;
}


NNLayer* Init_fc_lif_layer_17() {

	NNLayer* fc_lif_layer_17 = FC_LIF_Layer_Init(
		Getfc_lif_layer_17CMSPointer(),
		Getfc_lif_layer_17CMSLen(),
		Getfc_lif_layer_17WeightsPointer(),
		Getfc_lif_layer_17WeightsLen(),
		Getfc_lif_layer_17LIFParamPointer(),
		Getfc_lif_layer_17LIFParamLen(),
		Getfc_lif_layer_17LUTPointer(),
		Getfc_lif_layer_17LUTLen(),
		FC_LIF_LAYER_17_IS_LAST_LAYER,
		FC_LIF_LAYER_17_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_17_OUT_SPK_SCALE,
		FC_LIF_LAYER_17_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_17_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_17_TENSOR_ARENA_SIZE,
		fc_lif_layer_17_tensor_arena,
		fc_lif_layer_16_out_spk,
		fc_lif_layer_17_out_spk,
		FC_LIF_LAYER_17_BIAS_ADDR,
		FC_LIF_LAYER_17_WEIGHT_ADDR,
		FC_LIF_LAYER_17_V_MEM_ADDR,
		FC_LIF_LAYER_17_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_17_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_17_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_17_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_17_BIAS_LEN,
		FC_LIF_LAYER_17_WEIGHT_LEN,
		FC_LIF_LAYER_17_IN_SPK_SCALE,
		FC_LIF_LAYER_17_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_17_V_MEM_SCALE,
		FC_LIF_LAYER_17_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_17_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_17_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_17_OUT_SPK_SCALE,
		FC_LIF_LAYER_17_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_17;
}


NNLayer* Init_fc_lif_layer_18() {

	NNLayer* fc_lif_layer_18 = FC_LIF_Layer_Init(
		Getfc_lif_layer_18CMSPointer(),
		Getfc_lif_layer_18CMSLen(),
		Getfc_lif_layer_18WeightsPointer(),
		Getfc_lif_layer_18WeightsLen(),
		Getfc_lif_layer_18LIFParamPointer(),
		Getfc_lif_layer_18LIFParamLen(),
		Getfc_lif_layer_18LUTPointer(),
		Getfc_lif_layer_18LUTLen(),
		FC_LIF_LAYER_18_IS_LAST_LAYER,
		FC_LIF_LAYER_18_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_18_OUT_SPK_SCALE,
		FC_LIF_LAYER_18_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_18_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_18_TENSOR_ARENA_SIZE,
		fc_lif_layer_18_tensor_arena,
		fc_lif_layer_17_out_spk,
		fc_lif_layer_18_out_spk,
		FC_LIF_LAYER_18_BIAS_ADDR,
		FC_LIF_LAYER_18_WEIGHT_ADDR,
		FC_LIF_LAYER_18_V_MEM_ADDR,
		FC_LIF_LAYER_18_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_18_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_18_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_18_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_18_BIAS_LEN,
		FC_LIF_LAYER_18_WEIGHT_LEN,
		FC_LIF_LAYER_18_IN_SPK_SCALE,
		FC_LIF_LAYER_18_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_18_V_MEM_SCALE,
		FC_LIF_LAYER_18_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_18_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_18_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_18_OUT_SPK_SCALE,
		FC_LIF_LAYER_18_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_18;
}


NNLayer* Init_fc_lif_layer_19() {

	NNLayer* fc_lif_layer_19 = FC_LIF_Layer_Init(
		Getfc_lif_layer_19CMSPointer(),
		Getfc_lif_layer_19CMSLen(),
		Getfc_lif_layer_19WeightsPointer(),
		Getfc_lif_layer_19WeightsLen(),
		Getfc_lif_layer_19LIFParamPointer(),
		Getfc_lif_layer_19LIFParamLen(),
		Getfc_lif_layer_19LUTPointer(),
		Getfc_lif_layer_19LUTLen(),
		FC_LIF_LAYER_19_IS_LAST_LAYER,
		FC_LIF_LAYER_19_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_19_OUT_SPK_SCALE,
		FC_LIF_LAYER_19_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_19_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_19_TENSOR_ARENA_SIZE,
		fc_lif_layer_19_tensor_arena,
		fc_lif_layer_18_out_spk,
		fc_lif_layer_19_out_spk,
		FC_LIF_LAYER_19_BIAS_ADDR,
		FC_LIF_LAYER_19_WEIGHT_ADDR,
		FC_LIF_LAYER_19_V_MEM_ADDR,
		FC_LIF_LAYER_19_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_19_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_19_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_19_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_19_BIAS_LEN,
		FC_LIF_LAYER_19_WEIGHT_LEN,
		FC_LIF_LAYER_19_IN_SPK_SCALE,
		FC_LIF_LAYER_19_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_19_V_MEM_SCALE,
		FC_LIF_LAYER_19_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_19_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_19_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_19_OUT_SPK_SCALE,
		FC_LIF_LAYER_19_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_19;
}


NNLayer* Init_fc_lif_layer_20() {

	NNLayer* fc_lif_layer_20 = FC_LIF_Layer_Init(
		Getfc_lif_layer_20CMSPointer(),
		Getfc_lif_layer_20CMSLen(),
		Getfc_lif_layer_20WeightsPointer(),
		Getfc_lif_layer_20WeightsLen(),
		Getfc_lif_layer_20LIFParamPointer(),
		Getfc_lif_layer_20LIFParamLen(),
		Getfc_lif_layer_20LUTPointer(),
		Getfc_lif_layer_20LUTLen(),
		FC_LIF_LAYER_20_IS_LAST_LAYER,
		FC_LIF_LAYER_20_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_20_OUT_SPK_SCALE,
		FC_LIF_LAYER_20_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_20_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_20_TENSOR_ARENA_SIZE,
		fc_lif_layer_20_tensor_arena,
		fc_lif_layer_19_out_spk,
		fc_lif_layer_20_out_spk,
		FC_LIF_LAYER_20_BIAS_ADDR,
		FC_LIF_LAYER_20_WEIGHT_ADDR,
		FC_LIF_LAYER_20_V_MEM_ADDR,
		FC_LIF_LAYER_20_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_20_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_20_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_20_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_20_BIAS_LEN,
		FC_LIF_LAYER_20_WEIGHT_LEN,
		FC_LIF_LAYER_20_IN_SPK_SCALE,
		FC_LIF_LAYER_20_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_20_V_MEM_SCALE,
		FC_LIF_LAYER_20_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_20_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_20_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_20_OUT_SPK_SCALE,
		FC_LIF_LAYER_20_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_20;
}


NNLayer* Init_fc_lif_layer_21() {

	NNLayer* fc_lif_layer_21 = FC_LIF_Layer_Init(
		Getfc_lif_layer_21CMSPointer(),
		Getfc_lif_layer_21CMSLen(),
		Getfc_lif_layer_21WeightsPointer(),
		Getfc_lif_layer_21WeightsLen(),
		Getfc_lif_layer_21LIFParamPointer(),
		Getfc_lif_layer_21LIFParamLen(),
		Getfc_lif_layer_21LUTPointer(),
		Getfc_lif_layer_21LUTLen(),
		FC_LIF_LAYER_21_IS_LAST_LAYER,
		FC_LIF_LAYER_21_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_21_OUT_SPK_SCALE,
		FC_LIF_LAYER_21_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_21_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_21_TENSOR_ARENA_SIZE,
		fc_lif_layer_21_tensor_arena,
		fc_lif_layer_20_out_spk,
		fc_lif_layer_21_out_spk,
		FC_LIF_LAYER_21_BIAS_ADDR,
		FC_LIF_LAYER_21_WEIGHT_ADDR,
		FC_LIF_LAYER_21_V_MEM_ADDR,
		FC_LIF_LAYER_21_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_21_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_21_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_21_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_21_BIAS_LEN,
		FC_LIF_LAYER_21_WEIGHT_LEN,
		FC_LIF_LAYER_21_IN_SPK_SCALE,
		FC_LIF_LAYER_21_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_21_V_MEM_SCALE,
		FC_LIF_LAYER_21_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_21_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_21_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_21_OUT_SPK_SCALE,
		FC_LIF_LAYER_21_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_21;
}


NNLayer* Init_fc_lif_layer_22() {

	NNLayer* fc_lif_layer_22 = FC_LIF_Layer_Init(
		Getfc_lif_layer_22CMSPointer(),
		Getfc_lif_layer_22CMSLen(),
		Getfc_lif_layer_22WeightsPointer(),
		Getfc_lif_layer_22WeightsLen(),
		Getfc_lif_layer_22LIFParamPointer(),
		Getfc_lif_layer_22LIFParamLen(),
		Getfc_lif_layer_22LUTPointer(),
		Getfc_lif_layer_22LUTLen(),
		FC_LIF_LAYER_22_IS_LAST_LAYER,
		FC_LIF_LAYER_22_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_22_OUT_SPK_SCALE,
		FC_LIF_LAYER_22_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_22_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_22_TENSOR_ARENA_SIZE,
		fc_lif_layer_22_tensor_arena,
		fc_lif_layer_21_out_spk,
		fc_lif_layer_22_out_spk,
		FC_LIF_LAYER_22_BIAS_ADDR,
		FC_LIF_LAYER_22_WEIGHT_ADDR,
		FC_LIF_LAYER_22_V_MEM_ADDR,
		FC_LIF_LAYER_22_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_22_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_22_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_22_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_22_BIAS_LEN,
		FC_LIF_LAYER_22_WEIGHT_LEN,
		FC_LIF_LAYER_22_IN_SPK_SCALE,
		FC_LIF_LAYER_22_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_22_V_MEM_SCALE,
		FC_LIF_LAYER_22_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_22_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_22_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_22_OUT_SPK_SCALE,
		FC_LIF_LAYER_22_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_22;
}


NNLayer* Init_fc_lif_layer_23() {

	NNLayer* fc_lif_layer_23 = FC_LIF_Layer_Init(
		Getfc_lif_layer_23CMSPointer(),
		Getfc_lif_layer_23CMSLen(),
		Getfc_lif_layer_23WeightsPointer(),
		Getfc_lif_layer_23WeightsLen(),
		Getfc_lif_layer_23LIFParamPointer(),
		Getfc_lif_layer_23LIFParamLen(),
		Getfc_lif_layer_23LUTPointer(),
		Getfc_lif_layer_23LUTLen(),
		FC_LIF_LAYER_23_IS_LAST_LAYER,
		FC_LIF_LAYER_23_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_23_OUT_SPK_SCALE,
		FC_LIF_LAYER_23_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_23_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_23_TENSOR_ARENA_SIZE,
		fc_lif_layer_23_tensor_arena,
		fc_lif_layer_22_out_spk,
		fc_lif_layer_23_out_spk,
		FC_LIF_LAYER_23_BIAS_ADDR,
		FC_LIF_LAYER_23_WEIGHT_ADDR,
		FC_LIF_LAYER_23_V_MEM_ADDR,
		FC_LIF_LAYER_23_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_23_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_23_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_23_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_23_BIAS_LEN,
		FC_LIF_LAYER_23_WEIGHT_LEN,
		FC_LIF_LAYER_23_IN_SPK_SCALE,
		FC_LIF_LAYER_23_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_23_V_MEM_SCALE,
		FC_LIF_LAYER_23_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_23_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_23_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_23_OUT_SPK_SCALE,
		FC_LIF_LAYER_23_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_23;
}


NNLayer* Init_fc_lif_layer_24() {

	NNLayer* fc_lif_layer_24 = FC_LIF_Layer_Init(
		Getfc_lif_layer_24CMSPointer(),
		Getfc_lif_layer_24CMSLen(),
		Getfc_lif_layer_24WeightsPointer(),
		Getfc_lif_layer_24WeightsLen(),
		Getfc_lif_layer_24LIFParamPointer(),
		Getfc_lif_layer_24LIFParamLen(),
		Getfc_lif_layer_24LUTPointer(),
		Getfc_lif_layer_24LUTLen(),
		FC_LIF_LAYER_24_IS_LAST_LAYER,
		FC_LIF_LAYER_24_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_24_OUT_SPK_SCALE,
		FC_LIF_LAYER_24_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_24_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_24_TENSOR_ARENA_SIZE,
		fc_lif_layer_24_tensor_arena,
		fc_lif_layer_23_out_spk,
		fc_lif_layer_24_out_spk,
		FC_LIF_LAYER_24_BIAS_ADDR,
		FC_LIF_LAYER_24_WEIGHT_ADDR,
		FC_LIF_LAYER_24_V_MEM_ADDR,
		FC_LIF_LAYER_24_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_24_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_24_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_24_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_24_BIAS_LEN,
		FC_LIF_LAYER_24_WEIGHT_LEN,
		FC_LIF_LAYER_24_IN_SPK_SCALE,
		FC_LIF_LAYER_24_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_24_V_MEM_SCALE,
		FC_LIF_LAYER_24_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_24_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_24_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_24_OUT_SPK_SCALE,
		FC_LIF_LAYER_24_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_24;
}


NNLayer* Init_fc_lif_layer_25() {

	NNLayer* fc_lif_layer_25 = FC_LIF_Layer_Init(
		Getfc_lif_layer_25CMSPointer(),
		Getfc_lif_layer_25CMSLen(),
		Getfc_lif_layer_25WeightsPointer(),
		Getfc_lif_layer_25WeightsLen(),
		Getfc_lif_layer_25LIFParamPointer(),
		Getfc_lif_layer_25LIFParamLen(),
		Getfc_lif_layer_25LUTPointer(),
		Getfc_lif_layer_25LUTLen(),
		FC_LIF_LAYER_25_IS_LAST_LAYER,
		FC_LIF_LAYER_25_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_25_OUT_SPK_SCALE,
		FC_LIF_LAYER_25_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_25_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_25_TENSOR_ARENA_SIZE,
		fc_lif_layer_25_tensor_arena,
		fc_lif_layer_24_out_spk,
		fc_lif_layer_25_out_spk,
		FC_LIF_LAYER_25_BIAS_ADDR,
		FC_LIF_LAYER_25_WEIGHT_ADDR,
		FC_LIF_LAYER_25_V_MEM_ADDR,
		FC_LIF_LAYER_25_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_25_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_25_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_25_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_25_BIAS_LEN,
		FC_LIF_LAYER_25_WEIGHT_LEN,
		FC_LIF_LAYER_25_IN_SPK_SCALE,
		FC_LIF_LAYER_25_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_25_V_MEM_SCALE,
		FC_LIF_LAYER_25_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_25_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_25_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_25_OUT_SPK_SCALE,
		FC_LIF_LAYER_25_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_25;
}


NNLayer* Init_fc_lif_layer_26() {

	NNLayer* fc_lif_layer_26 = FC_LIF_Layer_Init(
		Getfc_lif_layer_26CMSPointer(),
		Getfc_lif_layer_26CMSLen(),
		Getfc_lif_layer_26WeightsPointer(),
		Getfc_lif_layer_26WeightsLen(),
		Getfc_lif_layer_26LIFParamPointer(),
		Getfc_lif_layer_26LIFParamLen(),
		Getfc_lif_layer_26LUTPointer(),
		Getfc_lif_layer_26LUTLen(),
		FC_LIF_LAYER_26_IS_LAST_LAYER,
		FC_LIF_LAYER_26_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_26_OUT_SPK_SCALE,
		FC_LIF_LAYER_26_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_26_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_26_TENSOR_ARENA_SIZE,
		fc_lif_layer_26_tensor_arena,
		fc_lif_layer_25_out_spk,
		fc_lif_layer_26_out_spk,
		FC_LIF_LAYER_26_BIAS_ADDR,
		FC_LIF_LAYER_26_WEIGHT_ADDR,
		FC_LIF_LAYER_26_V_MEM_ADDR,
		FC_LIF_LAYER_26_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_26_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_26_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_26_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_26_BIAS_LEN,
		FC_LIF_LAYER_26_WEIGHT_LEN,
		FC_LIF_LAYER_26_IN_SPK_SCALE,
		FC_LIF_LAYER_26_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_26_V_MEM_SCALE,
		FC_LIF_LAYER_26_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_26_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_26_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_26_OUT_SPK_SCALE,
		FC_LIF_LAYER_26_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_26;
}


NNLayer* Init_fc_lif_layer_27() {

	NNLayer* fc_lif_layer_27 = FC_LIF_Layer_Init(
		Getfc_lif_layer_27CMSPointer(),
		Getfc_lif_layer_27CMSLen(),
		Getfc_lif_layer_27WeightsPointer(),
		Getfc_lif_layer_27WeightsLen(),
		Getfc_lif_layer_27LIFParamPointer(),
		Getfc_lif_layer_27LIFParamLen(),
		Getfc_lif_layer_27LUTPointer(),
		Getfc_lif_layer_27LUTLen(),
		FC_LIF_LAYER_27_IS_LAST_LAYER,
		FC_LIF_LAYER_27_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_27_OUT_SPK_SCALE,
		FC_LIF_LAYER_27_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_27_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_27_TENSOR_ARENA_SIZE,
		fc_lif_layer_27_tensor_arena,
		fc_lif_layer_26_out_spk,
		fc_lif_layer_27_out_spk,
		FC_LIF_LAYER_27_BIAS_ADDR,
		FC_LIF_LAYER_27_WEIGHT_ADDR,
		FC_LIF_LAYER_27_V_MEM_ADDR,
		FC_LIF_LAYER_27_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_27_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_27_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_27_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_27_BIAS_LEN,
		FC_LIF_LAYER_27_WEIGHT_LEN,
		FC_LIF_LAYER_27_IN_SPK_SCALE,
		FC_LIF_LAYER_27_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_27_V_MEM_SCALE,
		FC_LIF_LAYER_27_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_27_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_27_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_27_OUT_SPK_SCALE,
		FC_LIF_LAYER_27_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_27;
}


NNLayer* Init_fc_lif_layer_28() {

	NNLayer* fc_lif_layer_28 = FC_LIF_Layer_Init(
		Getfc_lif_layer_28CMSPointer(),
		Getfc_lif_layer_28CMSLen(),
		Getfc_lif_layer_28WeightsPointer(),
		Getfc_lif_layer_28WeightsLen(),
		Getfc_lif_layer_28LIFParamPointer(),
		Getfc_lif_layer_28LIFParamLen(),
		Getfc_lif_layer_28LUTPointer(),
		Getfc_lif_layer_28LUTLen(),
		FC_LIF_LAYER_28_IS_LAST_LAYER,
		FC_LIF_LAYER_28_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_28_OUT_SPK_SCALE,
		FC_LIF_LAYER_28_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_28_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_28_TENSOR_ARENA_SIZE,
		fc_lif_layer_28_tensor_arena,
		fc_lif_layer_27_out_spk,
		fc_lif_layer_28_out_spk,
		FC_LIF_LAYER_28_BIAS_ADDR,
		FC_LIF_LAYER_28_WEIGHT_ADDR,
		FC_LIF_LAYER_28_V_MEM_ADDR,
		FC_LIF_LAYER_28_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_28_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_28_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_28_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_28_BIAS_LEN,
		FC_LIF_LAYER_28_WEIGHT_LEN,
		FC_LIF_LAYER_28_IN_SPK_SCALE,
		FC_LIF_LAYER_28_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_28_V_MEM_SCALE,
		FC_LIF_LAYER_28_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_28_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_28_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_28_OUT_SPK_SCALE,
		FC_LIF_LAYER_28_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_28;
}


NNLayer* Init_fc_lif_layer_29() {

	NNLayer* fc_lif_layer_29 = FC_LIF_Layer_Init(
		Getfc_lif_layer_29CMSPointer(),
		Getfc_lif_layer_29CMSLen(),
		Getfc_lif_layer_29WeightsPointer(),
		Getfc_lif_layer_29WeightsLen(),
		Getfc_lif_layer_29LIFParamPointer(),
		Getfc_lif_layer_29LIFParamLen(),
		Getfc_lif_layer_29LUTPointer(),
		Getfc_lif_layer_29LUTLen(),
		FC_LIF_LAYER_29_IS_LAST_LAYER,
		FC_LIF_LAYER_29_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_29_OUT_SPK_SCALE,
		FC_LIF_LAYER_29_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_29_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_29_TENSOR_ARENA_SIZE,
		fc_lif_layer_29_tensor_arena,
		fc_lif_layer_28_out_spk,
		fc_lif_layer_29_out_spk,
		FC_LIF_LAYER_29_BIAS_ADDR,
		FC_LIF_LAYER_29_WEIGHT_ADDR,
		FC_LIF_LAYER_29_V_MEM_ADDR,
		FC_LIF_LAYER_29_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_29_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_29_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_29_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_29_BIAS_LEN,
		FC_LIF_LAYER_29_WEIGHT_LEN,
		FC_LIF_LAYER_29_IN_SPK_SCALE,
		FC_LIF_LAYER_29_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_29_V_MEM_SCALE,
		FC_LIF_LAYER_29_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_29_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_29_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_29_OUT_SPK_SCALE,
		FC_LIF_LAYER_29_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_29;
}


NNLayer* Init_fc_lif_layer_30() {

	NNLayer* fc_lif_layer_30 = FC_LIF_Layer_Init(
		Getfc_lif_layer_30CMSPointer(),
		Getfc_lif_layer_30CMSLen(),
		Getfc_lif_layer_30WeightsPointer(),
		Getfc_lif_layer_30WeightsLen(),
		Getfc_lif_layer_30LIFParamPointer(),
		Getfc_lif_layer_30LIFParamLen(),
		Getfc_lif_layer_30LUTPointer(),
		Getfc_lif_layer_30LUTLen(),
		FC_LIF_LAYER_30_IS_LAST_LAYER,
		FC_LIF_LAYER_30_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_30_OUT_SPK_SCALE,
		FC_LIF_LAYER_30_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_30_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_30_TENSOR_ARENA_SIZE,
		fc_lif_layer_30_tensor_arena,
		fc_lif_layer_29_out_spk,
		fc_lif_layer_30_out_spk,
		FC_LIF_LAYER_30_BIAS_ADDR,
		FC_LIF_LAYER_30_WEIGHT_ADDR,
		FC_LIF_LAYER_30_V_MEM_ADDR,
		FC_LIF_LAYER_30_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_30_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_30_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_30_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_30_BIAS_LEN,
		FC_LIF_LAYER_30_WEIGHT_LEN,
		FC_LIF_LAYER_30_IN_SPK_SCALE,
		FC_LIF_LAYER_30_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_30_V_MEM_SCALE,
		FC_LIF_LAYER_30_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_30_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_30_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_30_OUT_SPK_SCALE,
		FC_LIF_LAYER_30_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_30;
}


NNLayer* Init_fc_lif_layer_31() {

	NNLayer* fc_lif_layer_31 = FC_LIF_Layer_Init(
		Getfc_lif_layer_31CMSPointer(),
		Getfc_lif_layer_31CMSLen(),
		Getfc_lif_layer_31WeightsPointer(),
		Getfc_lif_layer_31WeightsLen(),
		Getfc_lif_layer_31LIFParamPointer(),
		Getfc_lif_layer_31LIFParamLen(),
		Getfc_lif_layer_31LUTPointer(),
		Getfc_lif_layer_31LUTLen(),
		FC_LIF_LAYER_31_IS_LAST_LAYER,
		FC_LIF_LAYER_31_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_31_OUT_SPK_SCALE,
		FC_LIF_LAYER_31_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_31_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_31_TENSOR_ARENA_SIZE,
		fc_lif_layer_31_tensor_arena,
		fc_lif_layer_30_out_spk,
		fc_lif_layer_31_out_spk,
		FC_LIF_LAYER_31_BIAS_ADDR,
		FC_LIF_LAYER_31_WEIGHT_ADDR,
		FC_LIF_LAYER_31_V_MEM_ADDR,
		FC_LIF_LAYER_31_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_31_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_31_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_31_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_31_BIAS_LEN,
		FC_LIF_LAYER_31_WEIGHT_LEN,
		FC_LIF_LAYER_31_IN_SPK_SCALE,
		FC_LIF_LAYER_31_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_31_V_MEM_SCALE,
		FC_LIF_LAYER_31_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_31_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_31_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_31_OUT_SPK_SCALE,
		FC_LIF_LAYER_31_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_31;
}


NNLayer* Init_fc_lif_layer_32() {

	NNLayer* fc_lif_layer_32 = FC_LIF_Layer_Init(
		Getfc_lif_layer_32CMSPointer(),
		Getfc_lif_layer_32CMSLen(),
		Getfc_lif_layer_32WeightsPointer(),
		Getfc_lif_layer_32WeightsLen(),
		Getfc_lif_layer_32LIFParamPointer(),
		Getfc_lif_layer_32LIFParamLen(),
		Getfc_lif_layer_32LUTPointer(),
		Getfc_lif_layer_32LUTLen(),
		FC_LIF_LAYER_32_IS_LAST_LAYER,
		FC_LIF_LAYER_32_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_32_OUT_SPK_SCALE,
		FC_LIF_LAYER_32_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_32_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_32_TENSOR_ARENA_SIZE,
		fc_lif_layer_32_tensor_arena,
		fc_lif_layer_31_out_spk,
		fc_lif_layer_32_out_spk,
		FC_LIF_LAYER_32_BIAS_ADDR,
		FC_LIF_LAYER_32_WEIGHT_ADDR,
		FC_LIF_LAYER_32_V_MEM_ADDR,
		FC_LIF_LAYER_32_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_32_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_32_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_32_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_32_BIAS_LEN,
		FC_LIF_LAYER_32_WEIGHT_LEN,
		FC_LIF_LAYER_32_IN_SPK_SCALE,
		FC_LIF_LAYER_32_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_32_V_MEM_SCALE,
		FC_LIF_LAYER_32_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_32_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_32_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_32_OUT_SPK_SCALE,
		FC_LIF_LAYER_32_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_32;
}


NNLayer* Init_fc_lif_layer_33() {

	NNLayer* fc_lif_layer_33 = FC_LIF_Layer_Init(
		Getfc_lif_layer_33CMSPointer(),
		Getfc_lif_layer_33CMSLen(),
		Getfc_lif_layer_33WeightsPointer(),
		Getfc_lif_layer_33WeightsLen(),
		Getfc_lif_layer_33LIFParamPointer(),
		Getfc_lif_layer_33LIFParamLen(),
		Getfc_lif_layer_33LUTPointer(),
		Getfc_lif_layer_33LUTLen(),
		FC_LIF_LAYER_33_IS_LAST_LAYER,
		FC_LIF_LAYER_33_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_33_OUT_SPK_SCALE,
		FC_LIF_LAYER_33_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_33_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_33_TENSOR_ARENA_SIZE,
		fc_lif_layer_33_tensor_arena,
		fc_lif_layer_32_out_spk,
		fc_lif_layer_33_out_spk,
		FC_LIF_LAYER_33_BIAS_ADDR,
		FC_LIF_LAYER_33_WEIGHT_ADDR,
		FC_LIF_LAYER_33_V_MEM_ADDR,
		FC_LIF_LAYER_33_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_33_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_33_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_33_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_33_BIAS_LEN,
		FC_LIF_LAYER_33_WEIGHT_LEN,
		FC_LIF_LAYER_33_IN_SPK_SCALE,
		FC_LIF_LAYER_33_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_33_V_MEM_SCALE,
		FC_LIF_LAYER_33_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_33_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_33_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_33_OUT_SPK_SCALE,
		FC_LIF_LAYER_33_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_33;
}


NNLayer* Init_fc_lif_layer_34() {

	NNLayer* fc_lif_layer_34 = FC_LIF_Layer_Init(
		Getfc_lif_layer_34CMSPointer(),
		Getfc_lif_layer_34CMSLen(),
		Getfc_lif_layer_34WeightsPointer(),
		Getfc_lif_layer_34WeightsLen(),
		Getfc_lif_layer_34LIFParamPointer(),
		Getfc_lif_layer_34LIFParamLen(),
		Getfc_lif_layer_34LUTPointer(),
		Getfc_lif_layer_34LUTLen(),
		FC_LIF_LAYER_34_IS_LAST_LAYER,
		FC_LIF_LAYER_34_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_34_OUT_SPK_SCALE,
		FC_LIF_LAYER_34_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_34_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_34_TENSOR_ARENA_SIZE,
		fc_lif_layer_34_tensor_arena,
		fc_lif_layer_33_out_spk,
		fc_lif_layer_34_out_spk,
		FC_LIF_LAYER_34_BIAS_ADDR,
		FC_LIF_LAYER_34_WEIGHT_ADDR,
		FC_LIF_LAYER_34_V_MEM_ADDR,
		FC_LIF_LAYER_34_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_34_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_34_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_34_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_34_BIAS_LEN,
		FC_LIF_LAYER_34_WEIGHT_LEN,
		FC_LIF_LAYER_34_IN_SPK_SCALE,
		FC_LIF_LAYER_34_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_34_V_MEM_SCALE,
		FC_LIF_LAYER_34_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_34_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_34_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_34_OUT_SPK_SCALE,
		FC_LIF_LAYER_34_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_34;
}


NNLayer* Init_fc_lif_layer_35() {

	NNLayer* fc_lif_layer_35 = FC_LIF_Layer_Init(
		Getfc_lif_layer_35CMSPointer(),
		Getfc_lif_layer_35CMSLen(),
		Getfc_lif_layer_35WeightsPointer(),
		Getfc_lif_layer_35WeightsLen(),
		Getfc_lif_layer_35LIFParamPointer(),
		Getfc_lif_layer_35LIFParamLen(),
		Getfc_lif_layer_35LUTPointer(),
		Getfc_lif_layer_35LUTLen(),
		FC_LIF_LAYER_35_IS_LAST_LAYER,
		FC_LIF_LAYER_35_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_35_OUT_SPK_SCALE,
		FC_LIF_LAYER_35_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_35_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_35_TENSOR_ARENA_SIZE,
		fc_lif_layer_35_tensor_arena,
		fc_lif_layer_34_out_spk,
		fc_lif_layer_35_out_spk,
		FC_LIF_LAYER_35_BIAS_ADDR,
		FC_LIF_LAYER_35_WEIGHT_ADDR,
		FC_LIF_LAYER_35_V_MEM_ADDR,
		FC_LIF_LAYER_35_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_35_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_35_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_35_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_35_BIAS_LEN,
		FC_LIF_LAYER_35_WEIGHT_LEN,
		FC_LIF_LAYER_35_IN_SPK_SCALE,
		FC_LIF_LAYER_35_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_35_V_MEM_SCALE,
		FC_LIF_LAYER_35_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_35_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_35_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_35_OUT_SPK_SCALE,
		FC_LIF_LAYER_35_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_35;
}


NNLayer* Init_fc_lif_layer_36() {

	NNLayer* fc_lif_layer_36 = FC_LIF_Layer_Init(
		Getfc_lif_layer_36CMSPointer(),
		Getfc_lif_layer_36CMSLen(),
		Getfc_lif_layer_36WeightsPointer(),
		Getfc_lif_layer_36WeightsLen(),
		Getfc_lif_layer_36LIFParamPointer(),
		Getfc_lif_layer_36LIFParamLen(),
		Getfc_lif_layer_36LUTPointer(),
		Getfc_lif_layer_36LUTLen(),
		FC_LIF_LAYER_36_IS_LAST_LAYER,
		FC_LIF_LAYER_36_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_36_OUT_SPK_SCALE,
		FC_LIF_LAYER_36_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_36_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_36_TENSOR_ARENA_SIZE,
		fc_lif_layer_36_tensor_arena,
		fc_lif_layer_35_out_spk,
		fc_lif_layer_36_out_spk,
		FC_LIF_LAYER_36_BIAS_ADDR,
		FC_LIF_LAYER_36_WEIGHT_ADDR,
		FC_LIF_LAYER_36_V_MEM_ADDR,
		FC_LIF_LAYER_36_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_36_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_36_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_36_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_36_BIAS_LEN,
		FC_LIF_LAYER_36_WEIGHT_LEN,
		FC_LIF_LAYER_36_IN_SPK_SCALE,
		FC_LIF_LAYER_36_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_36_V_MEM_SCALE,
		FC_LIF_LAYER_36_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_36_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_36_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_36_OUT_SPK_SCALE,
		FC_LIF_LAYER_36_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_36;
}


NNLayer* Init_fc_lif_layer_37() {

	NNLayer* fc_lif_layer_37 = FC_LIF_Layer_Init(
		Getfc_lif_layer_37CMSPointer(),
		Getfc_lif_layer_37CMSLen(),
		Getfc_lif_layer_37WeightsPointer(),
		Getfc_lif_layer_37WeightsLen(),
		Getfc_lif_layer_37LIFParamPointer(),
		Getfc_lif_layer_37LIFParamLen(),
		Getfc_lif_layer_37LUTPointer(),
		Getfc_lif_layer_37LUTLen(),
		FC_LIF_LAYER_37_IS_LAST_LAYER,
		FC_LIF_LAYER_37_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_37_OUT_SPK_SCALE,
		FC_LIF_LAYER_37_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_37_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_37_TENSOR_ARENA_SIZE,
		fc_lif_layer_37_tensor_arena,
		fc_lif_layer_36_out_spk,
		fc_lif_layer_37_out_spk,
		FC_LIF_LAYER_37_BIAS_ADDR,
		FC_LIF_LAYER_37_WEIGHT_ADDR,
		FC_LIF_LAYER_37_V_MEM_ADDR,
		FC_LIF_LAYER_37_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_37_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_37_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_37_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_37_BIAS_LEN,
		FC_LIF_LAYER_37_WEIGHT_LEN,
		FC_LIF_LAYER_37_IN_SPK_SCALE,
		FC_LIF_LAYER_37_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_37_V_MEM_SCALE,
		FC_LIF_LAYER_37_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_37_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_37_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_37_OUT_SPK_SCALE,
		FC_LIF_LAYER_37_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_37;
}


NNLayer* Init_fc_lif_layer_38() {

	NNLayer* fc_lif_layer_38 = FC_LIF_Layer_Init(
		Getfc_lif_layer_38CMSPointer(),
		Getfc_lif_layer_38CMSLen(),
		Getfc_lif_layer_38WeightsPointer(),
		Getfc_lif_layer_38WeightsLen(),
		Getfc_lif_layer_38LIFParamPointer(),
		Getfc_lif_layer_38LIFParamLen(),
		Getfc_lif_layer_38LUTPointer(),
		Getfc_lif_layer_38LUTLen(),
		FC_LIF_LAYER_38_IS_LAST_LAYER,
		FC_LIF_LAYER_38_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_38_OUT_SPK_SCALE,
		FC_LIF_LAYER_38_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_38_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_38_TENSOR_ARENA_SIZE,
		fc_lif_layer_38_tensor_arena,
		fc_lif_layer_37_out_spk,
		fc_lif_layer_38_out_spk,
		FC_LIF_LAYER_38_BIAS_ADDR,
		FC_LIF_LAYER_38_WEIGHT_ADDR,
		FC_LIF_LAYER_38_V_MEM_ADDR,
		FC_LIF_LAYER_38_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_38_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_38_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_38_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_38_BIAS_LEN,
		FC_LIF_LAYER_38_WEIGHT_LEN,
		FC_LIF_LAYER_38_IN_SPK_SCALE,
		FC_LIF_LAYER_38_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_38_V_MEM_SCALE,
		FC_LIF_LAYER_38_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_38_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_38_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_38_OUT_SPK_SCALE,
		FC_LIF_LAYER_38_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_38;
}


NNLayer* Init_fc_lif_layer_39() {

	NNLayer* fc_lif_layer_39 = FC_LIF_Layer_Init(
		Getfc_lif_layer_39CMSPointer(),
		Getfc_lif_layer_39CMSLen(),
		Getfc_lif_layer_39WeightsPointer(),
		Getfc_lif_layer_39WeightsLen(),
		Getfc_lif_layer_39LIFParamPointer(),
		Getfc_lif_layer_39LIFParamLen(),
		Getfc_lif_layer_39LUTPointer(),
		Getfc_lif_layer_39LUTLen(),
		FC_LIF_LAYER_39_IS_LAST_LAYER,
		FC_LIF_LAYER_39_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_39_OUT_SPK_SCALE,
		FC_LIF_LAYER_39_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_39_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_39_TENSOR_ARENA_SIZE,
		fc_lif_layer_39_tensor_arena,
		fc_lif_layer_38_out_spk,
		fc_lif_layer_39_out_spk,
		FC_LIF_LAYER_39_BIAS_ADDR,
		FC_LIF_LAYER_39_WEIGHT_ADDR,
		FC_LIF_LAYER_39_V_MEM_ADDR,
		FC_LIF_LAYER_39_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_39_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_39_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_39_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_39_BIAS_LEN,
		FC_LIF_LAYER_39_WEIGHT_LEN,
		FC_LIF_LAYER_39_IN_SPK_SCALE,
		FC_LIF_LAYER_39_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_39_V_MEM_SCALE,
		FC_LIF_LAYER_39_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_39_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_39_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_39_OUT_SPK_SCALE,
		FC_LIF_LAYER_39_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_39;
}


NNLayer* Init_fc_lif_layer_40() {

	NNLayer* fc_lif_layer_40 = FC_LIF_Layer_Init(
		Getfc_lif_layer_40CMSPointer(),
		Getfc_lif_layer_40CMSLen(),
		Getfc_lif_layer_40WeightsPointer(),
		Getfc_lif_layer_40WeightsLen(),
		Getfc_lif_layer_40LIFParamPointer(),
		Getfc_lif_layer_40LIFParamLen(),
		Getfc_lif_layer_40LUTPointer(),
		Getfc_lif_layer_40LUTLen(),
		FC_LIF_LAYER_40_IS_LAST_LAYER,
		FC_LIF_LAYER_40_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_40_OUT_SPK_SCALE,
		FC_LIF_LAYER_40_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_40_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_40_TENSOR_ARENA_SIZE,
		fc_lif_layer_40_tensor_arena,
		fc_lif_layer_39_out_spk,
		fc_lif_layer_40_out_spk,
		FC_LIF_LAYER_40_BIAS_ADDR,
		FC_LIF_LAYER_40_WEIGHT_ADDR,
		FC_LIF_LAYER_40_V_MEM_ADDR,
		FC_LIF_LAYER_40_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_40_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_40_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_40_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_40_BIAS_LEN,
		FC_LIF_LAYER_40_WEIGHT_LEN,
		FC_LIF_LAYER_40_IN_SPK_SCALE,
		FC_LIF_LAYER_40_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_40_V_MEM_SCALE,
		FC_LIF_LAYER_40_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_40_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_40_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_40_OUT_SPK_SCALE,
		FC_LIF_LAYER_40_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_40;
}


NNLayer* Init_fc_lif_layer_41() {

	NNLayer* fc_lif_layer_41 = FC_LIF_Layer_Init(
		Getfc_lif_layer_41CMSPointer(),
		Getfc_lif_layer_41CMSLen(),
		Getfc_lif_layer_41WeightsPointer(),
		Getfc_lif_layer_41WeightsLen(),
		Getfc_lif_layer_41LIFParamPointer(),
		Getfc_lif_layer_41LIFParamLen(),
		Getfc_lif_layer_41LUTPointer(),
		Getfc_lif_layer_41LUTLen(),
		FC_LIF_LAYER_41_IS_LAST_LAYER,
		FC_LIF_LAYER_41_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_41_OUT_SPK_SCALE,
		FC_LIF_LAYER_41_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_41_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_41_TENSOR_ARENA_SIZE,
		fc_lif_layer_41_tensor_arena,
		fc_lif_layer_40_out_spk,
		fc_lif_layer_41_out_spk,
		FC_LIF_LAYER_41_BIAS_ADDR,
		FC_LIF_LAYER_41_WEIGHT_ADDR,
		FC_LIF_LAYER_41_V_MEM_ADDR,
		FC_LIF_LAYER_41_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_41_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_41_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_41_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_41_BIAS_LEN,
		FC_LIF_LAYER_41_WEIGHT_LEN,
		FC_LIF_LAYER_41_IN_SPK_SCALE,
		FC_LIF_LAYER_41_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_41_V_MEM_SCALE,
		FC_LIF_LAYER_41_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_41_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_41_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_41_OUT_SPK_SCALE,
		FC_LIF_LAYER_41_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_41;
}


NNLayer* Init_fc_lif_layer_42() {

	NNLayer* fc_lif_layer_42 = FC_LIF_Layer_Init(
		Getfc_lif_layer_42CMSPointer(),
		Getfc_lif_layer_42CMSLen(),
		Getfc_lif_layer_42WeightsPointer(),
		Getfc_lif_layer_42WeightsLen(),
		Getfc_lif_layer_42LIFParamPointer(),
		Getfc_lif_layer_42LIFParamLen(),
		Getfc_lif_layer_42LUTPointer(),
		Getfc_lif_layer_42LUTLen(),
		FC_LIF_LAYER_42_IS_LAST_LAYER,
		FC_LIF_LAYER_42_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_42_OUT_SPK_SCALE,
		FC_LIF_LAYER_42_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_42_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_42_TENSOR_ARENA_SIZE,
		fc_lif_layer_42_tensor_arena,
		fc_lif_layer_41_out_spk,
		fc_lif_layer_42_out_spk,
		FC_LIF_LAYER_42_BIAS_ADDR,
		FC_LIF_LAYER_42_WEIGHT_ADDR,
		FC_LIF_LAYER_42_V_MEM_ADDR,
		FC_LIF_LAYER_42_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_42_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_42_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_42_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_42_BIAS_LEN,
		FC_LIF_LAYER_42_WEIGHT_LEN,
		FC_LIF_LAYER_42_IN_SPK_SCALE,
		FC_LIF_LAYER_42_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_42_V_MEM_SCALE,
		FC_LIF_LAYER_42_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_42_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_42_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_42_OUT_SPK_SCALE,
		FC_LIF_LAYER_42_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_42;
}


NNLayer* Init_fc_lif_layer_43() {

	NNLayer* fc_lif_layer_43 = FC_LIF_Layer_Init(
		Getfc_lif_layer_43CMSPointer(),
		Getfc_lif_layer_43CMSLen(),
		Getfc_lif_layer_43WeightsPointer(),
		Getfc_lif_layer_43WeightsLen(),
		Getfc_lif_layer_43LIFParamPointer(),
		Getfc_lif_layer_43LIFParamLen(),
		Getfc_lif_layer_43LUTPointer(),
		Getfc_lif_layer_43LUTLen(),
		FC_LIF_LAYER_43_IS_LAST_LAYER,
		FC_LIF_LAYER_43_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_43_OUT_SPK_SCALE,
		FC_LIF_LAYER_43_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_43_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_43_TENSOR_ARENA_SIZE,
		fc_lif_layer_43_tensor_arena,
		fc_lif_layer_42_out_spk,
		fc_lif_layer_43_out_spk,
		FC_LIF_LAYER_43_BIAS_ADDR,
		FC_LIF_LAYER_43_WEIGHT_ADDR,
		FC_LIF_LAYER_43_V_MEM_ADDR,
		FC_LIF_LAYER_43_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_43_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_43_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_43_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_43_BIAS_LEN,
		FC_LIF_LAYER_43_WEIGHT_LEN,
		FC_LIF_LAYER_43_IN_SPK_SCALE,
		FC_LIF_LAYER_43_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_43_V_MEM_SCALE,
		FC_LIF_LAYER_43_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_43_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_43_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_43_OUT_SPK_SCALE,
		FC_LIF_LAYER_43_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_43;
}


NNLayer* Init_fc_lif_layer_44() {

	NNLayer* fc_lif_layer_44 = FC_LIF_Layer_Init(
		Getfc_lif_layer_44CMSPointer(),
		Getfc_lif_layer_44CMSLen(),
		Getfc_lif_layer_44WeightsPointer(),
		Getfc_lif_layer_44WeightsLen(),
		Getfc_lif_layer_44LIFParamPointer(),
		Getfc_lif_layer_44LIFParamLen(),
		Getfc_lif_layer_44LUTPointer(),
		Getfc_lif_layer_44LUTLen(),
		FC_LIF_LAYER_44_IS_LAST_LAYER,
		FC_LIF_LAYER_44_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_44_OUT_SPK_SCALE,
		FC_LIF_LAYER_44_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_44_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_44_TENSOR_ARENA_SIZE,
		fc_lif_layer_44_tensor_arena,
		fc_lif_layer_43_out_spk,
		fc_lif_layer_44_out_spk,
		FC_LIF_LAYER_44_BIAS_ADDR,
		FC_LIF_LAYER_44_WEIGHT_ADDR,
		FC_LIF_LAYER_44_V_MEM_ADDR,
		FC_LIF_LAYER_44_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_44_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_44_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_44_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_44_BIAS_LEN,
		FC_LIF_LAYER_44_WEIGHT_LEN,
		FC_LIF_LAYER_44_IN_SPK_SCALE,
		FC_LIF_LAYER_44_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_44_V_MEM_SCALE,
		FC_LIF_LAYER_44_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_44_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_44_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_44_OUT_SPK_SCALE,
		FC_LIF_LAYER_44_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_44;
}


NNLayer* Init_fc_lif_layer_45() {

	NNLayer* fc_lif_layer_45 = FC_LIF_Layer_Init(
		Getfc_lif_layer_45CMSPointer(),
		Getfc_lif_layer_45CMSLen(),
		Getfc_lif_layer_45WeightsPointer(),
		Getfc_lif_layer_45WeightsLen(),
		Getfc_lif_layer_45LIFParamPointer(),
		Getfc_lif_layer_45LIFParamLen(),
		Getfc_lif_layer_45LUTPointer(),
		Getfc_lif_layer_45LUTLen(),
		FC_LIF_LAYER_45_IS_LAST_LAYER,
		FC_LIF_LAYER_45_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_45_OUT_SPK_SCALE,
		FC_LIF_LAYER_45_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_45_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_45_TENSOR_ARENA_SIZE,
		fc_lif_layer_45_tensor_arena,
		fc_lif_layer_44_out_spk,
		fc_lif_layer_45_out_spk,
		FC_LIF_LAYER_45_BIAS_ADDR,
		FC_LIF_LAYER_45_WEIGHT_ADDR,
		FC_LIF_LAYER_45_V_MEM_ADDR,
		FC_LIF_LAYER_45_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_45_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_45_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_45_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_45_BIAS_LEN,
		FC_LIF_LAYER_45_WEIGHT_LEN,
		FC_LIF_LAYER_45_IN_SPK_SCALE,
		FC_LIF_LAYER_45_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_45_V_MEM_SCALE,
		FC_LIF_LAYER_45_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_45_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_45_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_45_OUT_SPK_SCALE,
		FC_LIF_LAYER_45_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_45;
}


NNLayer* Init_fc_lif_layer_46() {

	NNLayer* fc_lif_layer_46 = FC_LIF_Layer_Init(
		Getfc_lif_layer_46CMSPointer(),
		Getfc_lif_layer_46CMSLen(),
		Getfc_lif_layer_46WeightsPointer(),
		Getfc_lif_layer_46WeightsLen(),
		Getfc_lif_layer_46LIFParamPointer(),
		Getfc_lif_layer_46LIFParamLen(),
		Getfc_lif_layer_46LUTPointer(),
		Getfc_lif_layer_46LUTLen(),
		FC_LIF_LAYER_46_IS_LAST_LAYER,
		FC_LIF_LAYER_46_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_46_OUT_SPK_SCALE,
		FC_LIF_LAYER_46_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_46_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_46_TENSOR_ARENA_SIZE,
		fc_lif_layer_46_tensor_arena,
		fc_lif_layer_45_out_spk,
		fc_lif_layer_46_out_spk,
		FC_LIF_LAYER_46_BIAS_ADDR,
		FC_LIF_LAYER_46_WEIGHT_ADDR,
		FC_LIF_LAYER_46_V_MEM_ADDR,
		FC_LIF_LAYER_46_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_46_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_46_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_46_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_46_BIAS_LEN,
		FC_LIF_LAYER_46_WEIGHT_LEN,
		FC_LIF_LAYER_46_IN_SPK_SCALE,
		FC_LIF_LAYER_46_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_46_V_MEM_SCALE,
		FC_LIF_LAYER_46_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_46_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_46_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_46_OUT_SPK_SCALE,
		FC_LIF_LAYER_46_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_46;
}


NNLayer* Init_fc_lif_layer_47() {

	NNLayer* fc_lif_layer_47 = FC_LIF_Layer_Init(
		Getfc_lif_layer_47CMSPointer(),
		Getfc_lif_layer_47CMSLen(),
		Getfc_lif_layer_47WeightsPointer(),
		Getfc_lif_layer_47WeightsLen(),
		Getfc_lif_layer_47LIFParamPointer(),
		Getfc_lif_layer_47LIFParamLen(),
		Getfc_lif_layer_47LUTPointer(),
		Getfc_lif_layer_47LUTLen(),
		FC_LIF_LAYER_47_IS_LAST_LAYER,
		FC_LIF_LAYER_47_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_47_OUT_SPK_SCALE,
		FC_LIF_LAYER_47_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_47_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_47_TENSOR_ARENA_SIZE,
		fc_lif_layer_47_tensor_arena,
		fc_lif_layer_46_out_spk,
		fc_lif_layer_47_out_spk,
		FC_LIF_LAYER_47_BIAS_ADDR,
		FC_LIF_LAYER_47_WEIGHT_ADDR,
		FC_LIF_LAYER_47_V_MEM_ADDR,
		FC_LIF_LAYER_47_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_47_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_47_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_47_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_47_BIAS_LEN,
		FC_LIF_LAYER_47_WEIGHT_LEN,
		FC_LIF_LAYER_47_IN_SPK_SCALE,
		FC_LIF_LAYER_47_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_47_V_MEM_SCALE,
		FC_LIF_LAYER_47_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_47_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_47_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_47_OUT_SPK_SCALE,
		FC_LIF_LAYER_47_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_47;
}


NNLayer* Init_fc_lif_layer_48() {

	NNLayer* fc_lif_layer_48 = FC_LIF_Layer_Init(
		Getfc_lif_layer_48CMSPointer(),
		Getfc_lif_layer_48CMSLen(),
		Getfc_lif_layer_48WeightsPointer(),
		Getfc_lif_layer_48WeightsLen(),
		Getfc_lif_layer_48LIFParamPointer(),
		Getfc_lif_layer_48LIFParamLen(),
		Getfc_lif_layer_48LUTPointer(),
		Getfc_lif_layer_48LUTLen(),
		FC_LIF_LAYER_48_IS_LAST_LAYER,
		FC_LIF_LAYER_48_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_48_OUT_SPK_SCALE,
		FC_LIF_LAYER_48_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_48_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_48_TENSOR_ARENA_SIZE,
		fc_lif_layer_48_tensor_arena,
		fc_lif_layer_47_out_spk,
		fc_lif_layer_48_out_spk,
		FC_LIF_LAYER_48_BIAS_ADDR,
		FC_LIF_LAYER_48_WEIGHT_ADDR,
		FC_LIF_LAYER_48_V_MEM_ADDR,
		FC_LIF_LAYER_48_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_48_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_48_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_48_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_48_BIAS_LEN,
		FC_LIF_LAYER_48_WEIGHT_LEN,
		FC_LIF_LAYER_48_IN_SPK_SCALE,
		FC_LIF_LAYER_48_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_48_V_MEM_SCALE,
		FC_LIF_LAYER_48_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_48_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_48_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_48_OUT_SPK_SCALE,
		FC_LIF_LAYER_48_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_48;
}


NNLayer* Init_fc_lif_layer_49() {

	NNLayer* fc_lif_layer_49 = FC_LIF_Layer_Init(
		Getfc_lif_layer_49CMSPointer(),
		Getfc_lif_layer_49CMSLen(),
		Getfc_lif_layer_49WeightsPointer(),
		Getfc_lif_layer_49WeightsLen(),
		Getfc_lif_layer_49LIFParamPointer(),
		Getfc_lif_layer_49LIFParamLen(),
		Getfc_lif_layer_49LUTPointer(),
		Getfc_lif_layer_49LUTLen(),
		FC_LIF_LAYER_49_IS_LAST_LAYER,
		FC_LIF_LAYER_49_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_49_OUT_SPK_SCALE,
		FC_LIF_LAYER_49_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_49_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_49_TENSOR_ARENA_SIZE,
		fc_lif_layer_49_tensor_arena,
		fc_lif_layer_48_out_spk,
		fc_lif_layer_49_out_spk,
		FC_LIF_LAYER_49_BIAS_ADDR,
		FC_LIF_LAYER_49_WEIGHT_ADDR,
		FC_LIF_LAYER_49_V_MEM_ADDR,
		FC_LIF_LAYER_49_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_49_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_49_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_49_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_49_BIAS_LEN,
		FC_LIF_LAYER_49_WEIGHT_LEN,
		FC_LIF_LAYER_49_IN_SPK_SCALE,
		FC_LIF_LAYER_49_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_49_V_MEM_SCALE,
		FC_LIF_LAYER_49_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_49_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_49_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_49_OUT_SPK_SCALE,
		FC_LIF_LAYER_49_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_49;
}


NNLayer* Init_fc_lif_layer_50() {

	NNLayer* fc_lif_layer_50 = FC_LIF_Layer_Init(
		Getfc_lif_layer_50CMSPointer(),
		Getfc_lif_layer_50CMSLen(),
		Getfc_lif_layer_50WeightsPointer(),
		Getfc_lif_layer_50WeightsLen(),
		Getfc_lif_layer_50LIFParamPointer(),
		Getfc_lif_layer_50LIFParamLen(),
		Getfc_lif_layer_50LUTPointer(),
		Getfc_lif_layer_50LUTLen(),
		FC_LIF_LAYER_50_IS_LAST_LAYER,
		FC_LIF_LAYER_50_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_50_OUT_SPK_SCALE,
		FC_LIF_LAYER_50_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_50_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_50_TENSOR_ARENA_SIZE,
		fc_lif_layer_50_tensor_arena,
		fc_lif_layer_49_out_spk,
		fc_lif_layer_50_out_spk,
		FC_LIF_LAYER_50_BIAS_ADDR,
		FC_LIF_LAYER_50_WEIGHT_ADDR,
		FC_LIF_LAYER_50_V_MEM_ADDR,
		FC_LIF_LAYER_50_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_50_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_50_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_50_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_50_BIAS_LEN,
		FC_LIF_LAYER_50_WEIGHT_LEN,
		FC_LIF_LAYER_50_IN_SPK_SCALE,
		FC_LIF_LAYER_50_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_50_V_MEM_SCALE,
		FC_LIF_LAYER_50_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_50_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_50_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_50_OUT_SPK_SCALE,
		FC_LIF_LAYER_50_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_50;
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
	Init_fc_lif_layer_15,
	Init_fc_lif_layer_16,
	Init_fc_lif_layer_17,
	Init_fc_lif_layer_18,
	Init_fc_lif_layer_19,
	Init_fc_lif_layer_20,
	Init_fc_lif_layer_21,
	Init_fc_lif_layer_22,
	Init_fc_lif_layer_23,
	Init_fc_lif_layer_24,
	Init_fc_lif_layer_25,
	Init_fc_lif_layer_26,
	Init_fc_lif_layer_27,
	Init_fc_lif_layer_28,
	Init_fc_lif_layer_29,
	Init_fc_lif_layer_30,
	Init_fc_lif_layer_31,
	Init_fc_lif_layer_32,
	Init_fc_lif_layer_33,
	Init_fc_lif_layer_34,
	Init_fc_lif_layer_35,
	Init_fc_lif_layer_36,
	Init_fc_lif_layer_37,
	Init_fc_lif_layer_38,
	Init_fc_lif_layer_39,
	Init_fc_lif_layer_40,
	Init_fc_lif_layer_41,
	Init_fc_lif_layer_42,
	Init_fc_lif_layer_43,
	Init_fc_lif_layer_44,
	Init_fc_lif_layer_45,
	Init_fc_lif_layer_46,
	Init_fc_lif_layer_47,
	Init_fc_lif_layer_48,
	Init_fc_lif_layer_49,
	Init_fc_lif_layer_50,
};