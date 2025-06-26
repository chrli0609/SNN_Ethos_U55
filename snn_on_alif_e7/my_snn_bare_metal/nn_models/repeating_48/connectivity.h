#pragma once

#include "../include/nn_data_structure.h"
#include "../model.h"


#define MODEL_NAME "784x48^100x10"


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
#include "layers/fc_lif_layer_51.h"
#include "layers/fc_lif_layer_52.h"
#include "layers/fc_lif_layer_53.h"
#include "layers/fc_lif_layer_54.h"
#include "layers/fc_lif_layer_55.h"
#include "layers/fc_lif_layer_56.h"
#include "layers/fc_lif_layer_57.h"
#include "layers/fc_lif_layer_58.h"
#include "layers/fc_lif_layer_59.h"
#include "layers/fc_lif_layer_60.h"
#include "layers/fc_lif_layer_61.h"
#include "layers/fc_lif_layer_62.h"
#include "layers/fc_lif_layer_63.h"
#include "layers/fc_lif_layer_64.h"
#include "layers/fc_lif_layer_65.h"
#include "layers/fc_lif_layer_66.h"
#include "layers/fc_lif_layer_67.h"
#include "layers/fc_lif_layer_68.h"
#include "layers/fc_lif_layer_69.h"
#include "layers/fc_lif_layer_70.h"
#include "layers/fc_lif_layer_71.h"
#include "layers/fc_lif_layer_72.h"
#include "layers/fc_lif_layer_73.h"
#include "layers/fc_lif_layer_74.h"
#include "layers/fc_lif_layer_75.h"
#include "layers/fc_lif_layer_76.h"
#include "layers/fc_lif_layer_77.h"
#include "layers/fc_lif_layer_78.h"
#include "layers/fc_lif_layer_79.h"
#include "layers/fc_lif_layer_80.h"
#include "layers/fc_lif_layer_81.h"
#include "layers/fc_lif_layer_82.h"
#include "layers/fc_lif_layer_83.h"
#include "layers/fc_lif_layer_84.h"
#include "layers/fc_lif_layer_85.h"
#include "layers/fc_lif_layer_86.h"
#include "layers/fc_lif_layer_87.h"
#include "layers/fc_lif_layer_88.h"
#include "layers/fc_lif_layer_89.h"
#include "layers/fc_lif_layer_90.h"
#include "layers/fc_lif_layer_91.h"
#include "layers/fc_lif_layer_92.h"
#include "layers/fc_lif_layer_93.h"
#include "layers/fc_lif_layer_94.h"
#include "layers/fc_lif_layer_95.h"
#include "layers/fc_lif_layer_96.h"
#include "layers/fc_lif_layer_97.h"
#include "layers/fc_lif_layer_98.h"
#include "layers/fc_lif_layer_99.h"
#include "layers/fc_lif_layer_100.h"



#define MLP_INPUT_LAYER_SIZE	FC_LIF_LAYER_0_INPUT_LAYER_SIZE
#define MLP_OUTPUT_LAYER_SIZE	FC_LIF_LAYER_100_OUTPUT_LAYER_SIZE

#define MLP_NUM_LAYERS 101
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

static int8_t fc_lif_layer_51_tensor_arena[FC_LIF_LAYER_51_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_51_out_spk[FC_LIF_LAYER_51_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_52_tensor_arena[FC_LIF_LAYER_52_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_52_out_spk[FC_LIF_LAYER_52_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_53_tensor_arena[FC_LIF_LAYER_53_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_53_out_spk[FC_LIF_LAYER_53_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_54_tensor_arena[FC_LIF_LAYER_54_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_54_out_spk[FC_LIF_LAYER_54_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_55_tensor_arena[FC_LIF_LAYER_55_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_55_out_spk[FC_LIF_LAYER_55_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_56_tensor_arena[FC_LIF_LAYER_56_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_56_out_spk[FC_LIF_LAYER_56_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_57_tensor_arena[FC_LIF_LAYER_57_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_57_out_spk[FC_LIF_LAYER_57_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_58_tensor_arena[FC_LIF_LAYER_58_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_58_out_spk[FC_LIF_LAYER_58_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_59_tensor_arena[FC_LIF_LAYER_59_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_59_out_spk[FC_LIF_LAYER_59_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_60_tensor_arena[FC_LIF_LAYER_60_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_60_out_spk[FC_LIF_LAYER_60_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_61_tensor_arena[FC_LIF_LAYER_61_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_61_out_spk[FC_LIF_LAYER_61_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_62_tensor_arena[FC_LIF_LAYER_62_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_62_out_spk[FC_LIF_LAYER_62_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_63_tensor_arena[FC_LIF_LAYER_63_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_63_out_spk[FC_LIF_LAYER_63_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_64_tensor_arena[FC_LIF_LAYER_64_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_64_out_spk[FC_LIF_LAYER_64_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_65_tensor_arena[FC_LIF_LAYER_65_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_65_out_spk[FC_LIF_LAYER_65_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_66_tensor_arena[FC_LIF_LAYER_66_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_66_out_spk[FC_LIF_LAYER_66_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_67_tensor_arena[FC_LIF_LAYER_67_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_67_out_spk[FC_LIF_LAYER_67_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_68_tensor_arena[FC_LIF_LAYER_68_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_68_out_spk[FC_LIF_LAYER_68_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_69_tensor_arena[FC_LIF_LAYER_69_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_69_out_spk[FC_LIF_LAYER_69_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_70_tensor_arena[FC_LIF_LAYER_70_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_70_out_spk[FC_LIF_LAYER_70_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_71_tensor_arena[FC_LIF_LAYER_71_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_71_out_spk[FC_LIF_LAYER_71_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_72_tensor_arena[FC_LIF_LAYER_72_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_72_out_spk[FC_LIF_LAYER_72_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_73_tensor_arena[FC_LIF_LAYER_73_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_73_out_spk[FC_LIF_LAYER_73_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_74_tensor_arena[FC_LIF_LAYER_74_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_74_out_spk[FC_LIF_LAYER_74_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_75_tensor_arena[FC_LIF_LAYER_75_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_75_out_spk[FC_LIF_LAYER_75_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_76_tensor_arena[FC_LIF_LAYER_76_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_76_out_spk[FC_LIF_LAYER_76_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_77_tensor_arena[FC_LIF_LAYER_77_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_77_out_spk[FC_LIF_LAYER_77_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_78_tensor_arena[FC_LIF_LAYER_78_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_78_out_spk[FC_LIF_LAYER_78_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_79_tensor_arena[FC_LIF_LAYER_79_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_79_out_spk[FC_LIF_LAYER_79_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_80_tensor_arena[FC_LIF_LAYER_80_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_80_out_spk[FC_LIF_LAYER_80_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_81_tensor_arena[FC_LIF_LAYER_81_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_81_out_spk[FC_LIF_LAYER_81_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_82_tensor_arena[FC_LIF_LAYER_82_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_82_out_spk[FC_LIF_LAYER_82_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_83_tensor_arena[FC_LIF_LAYER_83_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_83_out_spk[FC_LIF_LAYER_83_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_84_tensor_arena[FC_LIF_LAYER_84_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_84_out_spk[FC_LIF_LAYER_84_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_85_tensor_arena[FC_LIF_LAYER_85_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_85_out_spk[FC_LIF_LAYER_85_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_86_tensor_arena[FC_LIF_LAYER_86_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_86_out_spk[FC_LIF_LAYER_86_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_87_tensor_arena[FC_LIF_LAYER_87_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_87_out_spk[FC_LIF_LAYER_87_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_88_tensor_arena[FC_LIF_LAYER_88_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_88_out_spk[FC_LIF_LAYER_88_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_89_tensor_arena[FC_LIF_LAYER_89_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_89_out_spk[FC_LIF_LAYER_89_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_90_tensor_arena[FC_LIF_LAYER_90_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_90_out_spk[FC_LIF_LAYER_90_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_91_tensor_arena[FC_LIF_LAYER_91_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_91_out_spk[FC_LIF_LAYER_91_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_92_tensor_arena[FC_LIF_LAYER_92_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_92_out_spk[FC_LIF_LAYER_92_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_93_tensor_arena[FC_LIF_LAYER_93_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_93_out_spk[FC_LIF_LAYER_93_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_94_tensor_arena[FC_LIF_LAYER_94_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_94_out_spk[FC_LIF_LAYER_94_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_95_tensor_arena[FC_LIF_LAYER_95_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_95_out_spk[FC_LIF_LAYER_95_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_96_tensor_arena[FC_LIF_LAYER_96_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_96_out_spk[FC_LIF_LAYER_96_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_97_tensor_arena[FC_LIF_LAYER_97_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_97_out_spk[FC_LIF_LAYER_97_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_98_tensor_arena[FC_LIF_LAYER_98_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_98_out_spk[FC_LIF_LAYER_98_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_99_tensor_arena[FC_LIF_LAYER_99_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_99_out_spk[FC_LIF_LAYER_99_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));

static int8_t fc_lif_layer_100_tensor_arena[FC_LIF_LAYER_100_TENSOR_ARENA_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));
static int8_t fc_lif_layer_100_out_spk[FC_LIF_LAYER_100_OUTPUT_LAYER_SIZE] __attribute__((section("model_params_sram1"))) __attribute__((aligned(16)));


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


NNLayer* Init_fc_lif_layer_51() {

	NNLayer* fc_lif_layer_51 = FC_LIF_Layer_Init(
		Getfc_lif_layer_51CMSPointer(),
		Getfc_lif_layer_51CMSLen(),
		Getfc_lif_layer_51WeightsPointer(),
		Getfc_lif_layer_51WeightsLen(),
		Getfc_lif_layer_51LIFParamPointer(),
		Getfc_lif_layer_51LIFParamLen(),
		Getfc_lif_layer_51LUTPointer(),
		Getfc_lif_layer_51LUTLen(),
		FC_LIF_LAYER_51_IS_LAST_LAYER,
		FC_LIF_LAYER_51_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_51_OUT_SPK_SCALE,
		FC_LIF_LAYER_51_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_51_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_51_TENSOR_ARENA_SIZE,
		fc_lif_layer_51_tensor_arena,
		fc_lif_layer_50_out_spk,
		fc_lif_layer_51_out_spk,
		FC_LIF_LAYER_51_BIAS_ADDR,
		FC_LIF_LAYER_51_WEIGHT_ADDR,
		FC_LIF_LAYER_51_V_MEM_ADDR,
		FC_LIF_LAYER_51_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_51_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_51_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_51_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_51_BIAS_LEN,
		FC_LIF_LAYER_51_WEIGHT_LEN,
		FC_LIF_LAYER_51_IN_SPK_SCALE,
		FC_LIF_LAYER_51_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_51_V_MEM_SCALE,
		FC_LIF_LAYER_51_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_51_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_51_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_51_OUT_SPK_SCALE,
		FC_LIF_LAYER_51_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_51;
}


NNLayer* Init_fc_lif_layer_52() {

	NNLayer* fc_lif_layer_52 = FC_LIF_Layer_Init(
		Getfc_lif_layer_52CMSPointer(),
		Getfc_lif_layer_52CMSLen(),
		Getfc_lif_layer_52WeightsPointer(),
		Getfc_lif_layer_52WeightsLen(),
		Getfc_lif_layer_52LIFParamPointer(),
		Getfc_lif_layer_52LIFParamLen(),
		Getfc_lif_layer_52LUTPointer(),
		Getfc_lif_layer_52LUTLen(),
		FC_LIF_LAYER_52_IS_LAST_LAYER,
		FC_LIF_LAYER_52_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_52_OUT_SPK_SCALE,
		FC_LIF_LAYER_52_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_52_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_52_TENSOR_ARENA_SIZE,
		fc_lif_layer_52_tensor_arena,
		fc_lif_layer_51_out_spk,
		fc_lif_layer_52_out_spk,
		FC_LIF_LAYER_52_BIAS_ADDR,
		FC_LIF_LAYER_52_WEIGHT_ADDR,
		FC_LIF_LAYER_52_V_MEM_ADDR,
		FC_LIF_LAYER_52_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_52_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_52_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_52_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_52_BIAS_LEN,
		FC_LIF_LAYER_52_WEIGHT_LEN,
		FC_LIF_LAYER_52_IN_SPK_SCALE,
		FC_LIF_LAYER_52_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_52_V_MEM_SCALE,
		FC_LIF_LAYER_52_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_52_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_52_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_52_OUT_SPK_SCALE,
		FC_LIF_LAYER_52_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_52;
}


NNLayer* Init_fc_lif_layer_53() {

	NNLayer* fc_lif_layer_53 = FC_LIF_Layer_Init(
		Getfc_lif_layer_53CMSPointer(),
		Getfc_lif_layer_53CMSLen(),
		Getfc_lif_layer_53WeightsPointer(),
		Getfc_lif_layer_53WeightsLen(),
		Getfc_lif_layer_53LIFParamPointer(),
		Getfc_lif_layer_53LIFParamLen(),
		Getfc_lif_layer_53LUTPointer(),
		Getfc_lif_layer_53LUTLen(),
		FC_LIF_LAYER_53_IS_LAST_LAYER,
		FC_LIF_LAYER_53_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_53_OUT_SPK_SCALE,
		FC_LIF_LAYER_53_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_53_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_53_TENSOR_ARENA_SIZE,
		fc_lif_layer_53_tensor_arena,
		fc_lif_layer_52_out_spk,
		fc_lif_layer_53_out_spk,
		FC_LIF_LAYER_53_BIAS_ADDR,
		FC_LIF_LAYER_53_WEIGHT_ADDR,
		FC_LIF_LAYER_53_V_MEM_ADDR,
		FC_LIF_LAYER_53_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_53_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_53_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_53_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_53_BIAS_LEN,
		FC_LIF_LAYER_53_WEIGHT_LEN,
		FC_LIF_LAYER_53_IN_SPK_SCALE,
		FC_LIF_LAYER_53_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_53_V_MEM_SCALE,
		FC_LIF_LAYER_53_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_53_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_53_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_53_OUT_SPK_SCALE,
		FC_LIF_LAYER_53_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_53;
}


NNLayer* Init_fc_lif_layer_54() {

	NNLayer* fc_lif_layer_54 = FC_LIF_Layer_Init(
		Getfc_lif_layer_54CMSPointer(),
		Getfc_lif_layer_54CMSLen(),
		Getfc_lif_layer_54WeightsPointer(),
		Getfc_lif_layer_54WeightsLen(),
		Getfc_lif_layer_54LIFParamPointer(),
		Getfc_lif_layer_54LIFParamLen(),
		Getfc_lif_layer_54LUTPointer(),
		Getfc_lif_layer_54LUTLen(),
		FC_LIF_LAYER_54_IS_LAST_LAYER,
		FC_LIF_LAYER_54_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_54_OUT_SPK_SCALE,
		FC_LIF_LAYER_54_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_54_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_54_TENSOR_ARENA_SIZE,
		fc_lif_layer_54_tensor_arena,
		fc_lif_layer_53_out_spk,
		fc_lif_layer_54_out_spk,
		FC_LIF_LAYER_54_BIAS_ADDR,
		FC_LIF_LAYER_54_WEIGHT_ADDR,
		FC_LIF_LAYER_54_V_MEM_ADDR,
		FC_LIF_LAYER_54_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_54_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_54_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_54_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_54_BIAS_LEN,
		FC_LIF_LAYER_54_WEIGHT_LEN,
		FC_LIF_LAYER_54_IN_SPK_SCALE,
		FC_LIF_LAYER_54_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_54_V_MEM_SCALE,
		FC_LIF_LAYER_54_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_54_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_54_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_54_OUT_SPK_SCALE,
		FC_LIF_LAYER_54_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_54;
}


NNLayer* Init_fc_lif_layer_55() {

	NNLayer* fc_lif_layer_55 = FC_LIF_Layer_Init(
		Getfc_lif_layer_55CMSPointer(),
		Getfc_lif_layer_55CMSLen(),
		Getfc_lif_layer_55WeightsPointer(),
		Getfc_lif_layer_55WeightsLen(),
		Getfc_lif_layer_55LIFParamPointer(),
		Getfc_lif_layer_55LIFParamLen(),
		Getfc_lif_layer_55LUTPointer(),
		Getfc_lif_layer_55LUTLen(),
		FC_LIF_LAYER_55_IS_LAST_LAYER,
		FC_LIF_LAYER_55_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_55_OUT_SPK_SCALE,
		FC_LIF_LAYER_55_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_55_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_55_TENSOR_ARENA_SIZE,
		fc_lif_layer_55_tensor_arena,
		fc_lif_layer_54_out_spk,
		fc_lif_layer_55_out_spk,
		FC_LIF_LAYER_55_BIAS_ADDR,
		FC_LIF_LAYER_55_WEIGHT_ADDR,
		FC_LIF_LAYER_55_V_MEM_ADDR,
		FC_LIF_LAYER_55_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_55_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_55_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_55_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_55_BIAS_LEN,
		FC_LIF_LAYER_55_WEIGHT_LEN,
		FC_LIF_LAYER_55_IN_SPK_SCALE,
		FC_LIF_LAYER_55_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_55_V_MEM_SCALE,
		FC_LIF_LAYER_55_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_55_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_55_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_55_OUT_SPK_SCALE,
		FC_LIF_LAYER_55_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_55;
}


NNLayer* Init_fc_lif_layer_56() {

	NNLayer* fc_lif_layer_56 = FC_LIF_Layer_Init(
		Getfc_lif_layer_56CMSPointer(),
		Getfc_lif_layer_56CMSLen(),
		Getfc_lif_layer_56WeightsPointer(),
		Getfc_lif_layer_56WeightsLen(),
		Getfc_lif_layer_56LIFParamPointer(),
		Getfc_lif_layer_56LIFParamLen(),
		Getfc_lif_layer_56LUTPointer(),
		Getfc_lif_layer_56LUTLen(),
		FC_LIF_LAYER_56_IS_LAST_LAYER,
		FC_LIF_LAYER_56_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_56_OUT_SPK_SCALE,
		FC_LIF_LAYER_56_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_56_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_56_TENSOR_ARENA_SIZE,
		fc_lif_layer_56_tensor_arena,
		fc_lif_layer_55_out_spk,
		fc_lif_layer_56_out_spk,
		FC_LIF_LAYER_56_BIAS_ADDR,
		FC_LIF_LAYER_56_WEIGHT_ADDR,
		FC_LIF_LAYER_56_V_MEM_ADDR,
		FC_LIF_LAYER_56_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_56_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_56_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_56_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_56_BIAS_LEN,
		FC_LIF_LAYER_56_WEIGHT_LEN,
		FC_LIF_LAYER_56_IN_SPK_SCALE,
		FC_LIF_LAYER_56_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_56_V_MEM_SCALE,
		FC_LIF_LAYER_56_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_56_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_56_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_56_OUT_SPK_SCALE,
		FC_LIF_LAYER_56_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_56;
}


NNLayer* Init_fc_lif_layer_57() {

	NNLayer* fc_lif_layer_57 = FC_LIF_Layer_Init(
		Getfc_lif_layer_57CMSPointer(),
		Getfc_lif_layer_57CMSLen(),
		Getfc_lif_layer_57WeightsPointer(),
		Getfc_lif_layer_57WeightsLen(),
		Getfc_lif_layer_57LIFParamPointer(),
		Getfc_lif_layer_57LIFParamLen(),
		Getfc_lif_layer_57LUTPointer(),
		Getfc_lif_layer_57LUTLen(),
		FC_LIF_LAYER_57_IS_LAST_LAYER,
		FC_LIF_LAYER_57_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_57_OUT_SPK_SCALE,
		FC_LIF_LAYER_57_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_57_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_57_TENSOR_ARENA_SIZE,
		fc_lif_layer_57_tensor_arena,
		fc_lif_layer_56_out_spk,
		fc_lif_layer_57_out_spk,
		FC_LIF_LAYER_57_BIAS_ADDR,
		FC_LIF_LAYER_57_WEIGHT_ADDR,
		FC_LIF_LAYER_57_V_MEM_ADDR,
		FC_LIF_LAYER_57_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_57_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_57_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_57_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_57_BIAS_LEN,
		FC_LIF_LAYER_57_WEIGHT_LEN,
		FC_LIF_LAYER_57_IN_SPK_SCALE,
		FC_LIF_LAYER_57_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_57_V_MEM_SCALE,
		FC_LIF_LAYER_57_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_57_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_57_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_57_OUT_SPK_SCALE,
		FC_LIF_LAYER_57_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_57;
}


NNLayer* Init_fc_lif_layer_58() {

	NNLayer* fc_lif_layer_58 = FC_LIF_Layer_Init(
		Getfc_lif_layer_58CMSPointer(),
		Getfc_lif_layer_58CMSLen(),
		Getfc_lif_layer_58WeightsPointer(),
		Getfc_lif_layer_58WeightsLen(),
		Getfc_lif_layer_58LIFParamPointer(),
		Getfc_lif_layer_58LIFParamLen(),
		Getfc_lif_layer_58LUTPointer(),
		Getfc_lif_layer_58LUTLen(),
		FC_LIF_LAYER_58_IS_LAST_LAYER,
		FC_LIF_LAYER_58_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_58_OUT_SPK_SCALE,
		FC_LIF_LAYER_58_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_58_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_58_TENSOR_ARENA_SIZE,
		fc_lif_layer_58_tensor_arena,
		fc_lif_layer_57_out_spk,
		fc_lif_layer_58_out_spk,
		FC_LIF_LAYER_58_BIAS_ADDR,
		FC_LIF_LAYER_58_WEIGHT_ADDR,
		FC_LIF_LAYER_58_V_MEM_ADDR,
		FC_LIF_LAYER_58_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_58_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_58_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_58_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_58_BIAS_LEN,
		FC_LIF_LAYER_58_WEIGHT_LEN,
		FC_LIF_LAYER_58_IN_SPK_SCALE,
		FC_LIF_LAYER_58_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_58_V_MEM_SCALE,
		FC_LIF_LAYER_58_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_58_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_58_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_58_OUT_SPK_SCALE,
		FC_LIF_LAYER_58_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_58;
}


NNLayer* Init_fc_lif_layer_59() {

	NNLayer* fc_lif_layer_59 = FC_LIF_Layer_Init(
		Getfc_lif_layer_59CMSPointer(),
		Getfc_lif_layer_59CMSLen(),
		Getfc_lif_layer_59WeightsPointer(),
		Getfc_lif_layer_59WeightsLen(),
		Getfc_lif_layer_59LIFParamPointer(),
		Getfc_lif_layer_59LIFParamLen(),
		Getfc_lif_layer_59LUTPointer(),
		Getfc_lif_layer_59LUTLen(),
		FC_LIF_LAYER_59_IS_LAST_LAYER,
		FC_LIF_LAYER_59_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_59_OUT_SPK_SCALE,
		FC_LIF_LAYER_59_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_59_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_59_TENSOR_ARENA_SIZE,
		fc_lif_layer_59_tensor_arena,
		fc_lif_layer_58_out_spk,
		fc_lif_layer_59_out_spk,
		FC_LIF_LAYER_59_BIAS_ADDR,
		FC_LIF_LAYER_59_WEIGHT_ADDR,
		FC_LIF_LAYER_59_V_MEM_ADDR,
		FC_LIF_LAYER_59_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_59_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_59_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_59_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_59_BIAS_LEN,
		FC_LIF_LAYER_59_WEIGHT_LEN,
		FC_LIF_LAYER_59_IN_SPK_SCALE,
		FC_LIF_LAYER_59_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_59_V_MEM_SCALE,
		FC_LIF_LAYER_59_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_59_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_59_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_59_OUT_SPK_SCALE,
		FC_LIF_LAYER_59_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_59;
}


NNLayer* Init_fc_lif_layer_60() {

	NNLayer* fc_lif_layer_60 = FC_LIF_Layer_Init(
		Getfc_lif_layer_60CMSPointer(),
		Getfc_lif_layer_60CMSLen(),
		Getfc_lif_layer_60WeightsPointer(),
		Getfc_lif_layer_60WeightsLen(),
		Getfc_lif_layer_60LIFParamPointer(),
		Getfc_lif_layer_60LIFParamLen(),
		Getfc_lif_layer_60LUTPointer(),
		Getfc_lif_layer_60LUTLen(),
		FC_LIF_LAYER_60_IS_LAST_LAYER,
		FC_LIF_LAYER_60_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_60_OUT_SPK_SCALE,
		FC_LIF_LAYER_60_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_60_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_60_TENSOR_ARENA_SIZE,
		fc_lif_layer_60_tensor_arena,
		fc_lif_layer_59_out_spk,
		fc_lif_layer_60_out_spk,
		FC_LIF_LAYER_60_BIAS_ADDR,
		FC_LIF_LAYER_60_WEIGHT_ADDR,
		FC_LIF_LAYER_60_V_MEM_ADDR,
		FC_LIF_LAYER_60_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_60_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_60_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_60_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_60_BIAS_LEN,
		FC_LIF_LAYER_60_WEIGHT_LEN,
		FC_LIF_LAYER_60_IN_SPK_SCALE,
		FC_LIF_LAYER_60_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_60_V_MEM_SCALE,
		FC_LIF_LAYER_60_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_60_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_60_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_60_OUT_SPK_SCALE,
		FC_LIF_LAYER_60_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_60;
}


NNLayer* Init_fc_lif_layer_61() {

	NNLayer* fc_lif_layer_61 = FC_LIF_Layer_Init(
		Getfc_lif_layer_61CMSPointer(),
		Getfc_lif_layer_61CMSLen(),
		Getfc_lif_layer_61WeightsPointer(),
		Getfc_lif_layer_61WeightsLen(),
		Getfc_lif_layer_61LIFParamPointer(),
		Getfc_lif_layer_61LIFParamLen(),
		Getfc_lif_layer_61LUTPointer(),
		Getfc_lif_layer_61LUTLen(),
		FC_LIF_LAYER_61_IS_LAST_LAYER,
		FC_LIF_LAYER_61_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_61_OUT_SPK_SCALE,
		FC_LIF_LAYER_61_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_61_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_61_TENSOR_ARENA_SIZE,
		fc_lif_layer_61_tensor_arena,
		fc_lif_layer_60_out_spk,
		fc_lif_layer_61_out_spk,
		FC_LIF_LAYER_61_BIAS_ADDR,
		FC_LIF_LAYER_61_WEIGHT_ADDR,
		FC_LIF_LAYER_61_V_MEM_ADDR,
		FC_LIF_LAYER_61_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_61_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_61_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_61_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_61_BIAS_LEN,
		FC_LIF_LAYER_61_WEIGHT_LEN,
		FC_LIF_LAYER_61_IN_SPK_SCALE,
		FC_LIF_LAYER_61_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_61_V_MEM_SCALE,
		FC_LIF_LAYER_61_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_61_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_61_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_61_OUT_SPK_SCALE,
		FC_LIF_LAYER_61_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_61;
}


NNLayer* Init_fc_lif_layer_62() {

	NNLayer* fc_lif_layer_62 = FC_LIF_Layer_Init(
		Getfc_lif_layer_62CMSPointer(),
		Getfc_lif_layer_62CMSLen(),
		Getfc_lif_layer_62WeightsPointer(),
		Getfc_lif_layer_62WeightsLen(),
		Getfc_lif_layer_62LIFParamPointer(),
		Getfc_lif_layer_62LIFParamLen(),
		Getfc_lif_layer_62LUTPointer(),
		Getfc_lif_layer_62LUTLen(),
		FC_LIF_LAYER_62_IS_LAST_LAYER,
		FC_LIF_LAYER_62_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_62_OUT_SPK_SCALE,
		FC_LIF_LAYER_62_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_62_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_62_TENSOR_ARENA_SIZE,
		fc_lif_layer_62_tensor_arena,
		fc_lif_layer_61_out_spk,
		fc_lif_layer_62_out_spk,
		FC_LIF_LAYER_62_BIAS_ADDR,
		FC_LIF_LAYER_62_WEIGHT_ADDR,
		FC_LIF_LAYER_62_V_MEM_ADDR,
		FC_LIF_LAYER_62_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_62_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_62_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_62_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_62_BIAS_LEN,
		FC_LIF_LAYER_62_WEIGHT_LEN,
		FC_LIF_LAYER_62_IN_SPK_SCALE,
		FC_LIF_LAYER_62_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_62_V_MEM_SCALE,
		FC_LIF_LAYER_62_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_62_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_62_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_62_OUT_SPK_SCALE,
		FC_LIF_LAYER_62_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_62;
}


NNLayer* Init_fc_lif_layer_63() {

	NNLayer* fc_lif_layer_63 = FC_LIF_Layer_Init(
		Getfc_lif_layer_63CMSPointer(),
		Getfc_lif_layer_63CMSLen(),
		Getfc_lif_layer_63WeightsPointer(),
		Getfc_lif_layer_63WeightsLen(),
		Getfc_lif_layer_63LIFParamPointer(),
		Getfc_lif_layer_63LIFParamLen(),
		Getfc_lif_layer_63LUTPointer(),
		Getfc_lif_layer_63LUTLen(),
		FC_LIF_LAYER_63_IS_LAST_LAYER,
		FC_LIF_LAYER_63_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_63_OUT_SPK_SCALE,
		FC_LIF_LAYER_63_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_63_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_63_TENSOR_ARENA_SIZE,
		fc_lif_layer_63_tensor_arena,
		fc_lif_layer_62_out_spk,
		fc_lif_layer_63_out_spk,
		FC_LIF_LAYER_63_BIAS_ADDR,
		FC_LIF_LAYER_63_WEIGHT_ADDR,
		FC_LIF_LAYER_63_V_MEM_ADDR,
		FC_LIF_LAYER_63_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_63_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_63_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_63_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_63_BIAS_LEN,
		FC_LIF_LAYER_63_WEIGHT_LEN,
		FC_LIF_LAYER_63_IN_SPK_SCALE,
		FC_LIF_LAYER_63_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_63_V_MEM_SCALE,
		FC_LIF_LAYER_63_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_63_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_63_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_63_OUT_SPK_SCALE,
		FC_LIF_LAYER_63_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_63;
}


NNLayer* Init_fc_lif_layer_64() {

	NNLayer* fc_lif_layer_64 = FC_LIF_Layer_Init(
		Getfc_lif_layer_64CMSPointer(),
		Getfc_lif_layer_64CMSLen(),
		Getfc_lif_layer_64WeightsPointer(),
		Getfc_lif_layer_64WeightsLen(),
		Getfc_lif_layer_64LIFParamPointer(),
		Getfc_lif_layer_64LIFParamLen(),
		Getfc_lif_layer_64LUTPointer(),
		Getfc_lif_layer_64LUTLen(),
		FC_LIF_LAYER_64_IS_LAST_LAYER,
		FC_LIF_LAYER_64_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_64_OUT_SPK_SCALE,
		FC_LIF_LAYER_64_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_64_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_64_TENSOR_ARENA_SIZE,
		fc_lif_layer_64_tensor_arena,
		fc_lif_layer_63_out_spk,
		fc_lif_layer_64_out_spk,
		FC_LIF_LAYER_64_BIAS_ADDR,
		FC_LIF_LAYER_64_WEIGHT_ADDR,
		FC_LIF_LAYER_64_V_MEM_ADDR,
		FC_LIF_LAYER_64_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_64_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_64_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_64_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_64_BIAS_LEN,
		FC_LIF_LAYER_64_WEIGHT_LEN,
		FC_LIF_LAYER_64_IN_SPK_SCALE,
		FC_LIF_LAYER_64_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_64_V_MEM_SCALE,
		FC_LIF_LAYER_64_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_64_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_64_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_64_OUT_SPK_SCALE,
		FC_LIF_LAYER_64_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_64;
}


NNLayer* Init_fc_lif_layer_65() {

	NNLayer* fc_lif_layer_65 = FC_LIF_Layer_Init(
		Getfc_lif_layer_65CMSPointer(),
		Getfc_lif_layer_65CMSLen(),
		Getfc_lif_layer_65WeightsPointer(),
		Getfc_lif_layer_65WeightsLen(),
		Getfc_lif_layer_65LIFParamPointer(),
		Getfc_lif_layer_65LIFParamLen(),
		Getfc_lif_layer_65LUTPointer(),
		Getfc_lif_layer_65LUTLen(),
		FC_LIF_LAYER_65_IS_LAST_LAYER,
		FC_LIF_LAYER_65_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_65_OUT_SPK_SCALE,
		FC_LIF_LAYER_65_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_65_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_65_TENSOR_ARENA_SIZE,
		fc_lif_layer_65_tensor_arena,
		fc_lif_layer_64_out_spk,
		fc_lif_layer_65_out_spk,
		FC_LIF_LAYER_65_BIAS_ADDR,
		FC_LIF_LAYER_65_WEIGHT_ADDR,
		FC_LIF_LAYER_65_V_MEM_ADDR,
		FC_LIF_LAYER_65_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_65_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_65_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_65_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_65_BIAS_LEN,
		FC_LIF_LAYER_65_WEIGHT_LEN,
		FC_LIF_LAYER_65_IN_SPK_SCALE,
		FC_LIF_LAYER_65_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_65_V_MEM_SCALE,
		FC_LIF_LAYER_65_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_65_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_65_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_65_OUT_SPK_SCALE,
		FC_LIF_LAYER_65_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_65;
}


NNLayer* Init_fc_lif_layer_66() {

	NNLayer* fc_lif_layer_66 = FC_LIF_Layer_Init(
		Getfc_lif_layer_66CMSPointer(),
		Getfc_lif_layer_66CMSLen(),
		Getfc_lif_layer_66WeightsPointer(),
		Getfc_lif_layer_66WeightsLen(),
		Getfc_lif_layer_66LIFParamPointer(),
		Getfc_lif_layer_66LIFParamLen(),
		Getfc_lif_layer_66LUTPointer(),
		Getfc_lif_layer_66LUTLen(),
		FC_LIF_LAYER_66_IS_LAST_LAYER,
		FC_LIF_LAYER_66_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_66_OUT_SPK_SCALE,
		FC_LIF_LAYER_66_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_66_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_66_TENSOR_ARENA_SIZE,
		fc_lif_layer_66_tensor_arena,
		fc_lif_layer_65_out_spk,
		fc_lif_layer_66_out_spk,
		FC_LIF_LAYER_66_BIAS_ADDR,
		FC_LIF_LAYER_66_WEIGHT_ADDR,
		FC_LIF_LAYER_66_V_MEM_ADDR,
		FC_LIF_LAYER_66_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_66_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_66_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_66_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_66_BIAS_LEN,
		FC_LIF_LAYER_66_WEIGHT_LEN,
		FC_LIF_LAYER_66_IN_SPK_SCALE,
		FC_LIF_LAYER_66_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_66_V_MEM_SCALE,
		FC_LIF_LAYER_66_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_66_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_66_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_66_OUT_SPK_SCALE,
		FC_LIF_LAYER_66_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_66;
}


NNLayer* Init_fc_lif_layer_67() {

	NNLayer* fc_lif_layer_67 = FC_LIF_Layer_Init(
		Getfc_lif_layer_67CMSPointer(),
		Getfc_lif_layer_67CMSLen(),
		Getfc_lif_layer_67WeightsPointer(),
		Getfc_lif_layer_67WeightsLen(),
		Getfc_lif_layer_67LIFParamPointer(),
		Getfc_lif_layer_67LIFParamLen(),
		Getfc_lif_layer_67LUTPointer(),
		Getfc_lif_layer_67LUTLen(),
		FC_LIF_LAYER_67_IS_LAST_LAYER,
		FC_LIF_LAYER_67_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_67_OUT_SPK_SCALE,
		FC_LIF_LAYER_67_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_67_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_67_TENSOR_ARENA_SIZE,
		fc_lif_layer_67_tensor_arena,
		fc_lif_layer_66_out_spk,
		fc_lif_layer_67_out_spk,
		FC_LIF_LAYER_67_BIAS_ADDR,
		FC_LIF_LAYER_67_WEIGHT_ADDR,
		FC_LIF_LAYER_67_V_MEM_ADDR,
		FC_LIF_LAYER_67_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_67_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_67_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_67_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_67_BIAS_LEN,
		FC_LIF_LAYER_67_WEIGHT_LEN,
		FC_LIF_LAYER_67_IN_SPK_SCALE,
		FC_LIF_LAYER_67_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_67_V_MEM_SCALE,
		FC_LIF_LAYER_67_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_67_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_67_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_67_OUT_SPK_SCALE,
		FC_LIF_LAYER_67_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_67;
}


NNLayer* Init_fc_lif_layer_68() {

	NNLayer* fc_lif_layer_68 = FC_LIF_Layer_Init(
		Getfc_lif_layer_68CMSPointer(),
		Getfc_lif_layer_68CMSLen(),
		Getfc_lif_layer_68WeightsPointer(),
		Getfc_lif_layer_68WeightsLen(),
		Getfc_lif_layer_68LIFParamPointer(),
		Getfc_lif_layer_68LIFParamLen(),
		Getfc_lif_layer_68LUTPointer(),
		Getfc_lif_layer_68LUTLen(),
		FC_LIF_LAYER_68_IS_LAST_LAYER,
		FC_LIF_LAYER_68_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_68_OUT_SPK_SCALE,
		FC_LIF_LAYER_68_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_68_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_68_TENSOR_ARENA_SIZE,
		fc_lif_layer_68_tensor_arena,
		fc_lif_layer_67_out_spk,
		fc_lif_layer_68_out_spk,
		FC_LIF_LAYER_68_BIAS_ADDR,
		FC_LIF_LAYER_68_WEIGHT_ADDR,
		FC_LIF_LAYER_68_V_MEM_ADDR,
		FC_LIF_LAYER_68_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_68_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_68_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_68_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_68_BIAS_LEN,
		FC_LIF_LAYER_68_WEIGHT_LEN,
		FC_LIF_LAYER_68_IN_SPK_SCALE,
		FC_LIF_LAYER_68_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_68_V_MEM_SCALE,
		FC_LIF_LAYER_68_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_68_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_68_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_68_OUT_SPK_SCALE,
		FC_LIF_LAYER_68_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_68;
}


NNLayer* Init_fc_lif_layer_69() {

	NNLayer* fc_lif_layer_69 = FC_LIF_Layer_Init(
		Getfc_lif_layer_69CMSPointer(),
		Getfc_lif_layer_69CMSLen(),
		Getfc_lif_layer_69WeightsPointer(),
		Getfc_lif_layer_69WeightsLen(),
		Getfc_lif_layer_69LIFParamPointer(),
		Getfc_lif_layer_69LIFParamLen(),
		Getfc_lif_layer_69LUTPointer(),
		Getfc_lif_layer_69LUTLen(),
		FC_LIF_LAYER_69_IS_LAST_LAYER,
		FC_LIF_LAYER_69_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_69_OUT_SPK_SCALE,
		FC_LIF_LAYER_69_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_69_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_69_TENSOR_ARENA_SIZE,
		fc_lif_layer_69_tensor_arena,
		fc_lif_layer_68_out_spk,
		fc_lif_layer_69_out_spk,
		FC_LIF_LAYER_69_BIAS_ADDR,
		FC_LIF_LAYER_69_WEIGHT_ADDR,
		FC_LIF_LAYER_69_V_MEM_ADDR,
		FC_LIF_LAYER_69_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_69_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_69_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_69_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_69_BIAS_LEN,
		FC_LIF_LAYER_69_WEIGHT_LEN,
		FC_LIF_LAYER_69_IN_SPK_SCALE,
		FC_LIF_LAYER_69_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_69_V_MEM_SCALE,
		FC_LIF_LAYER_69_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_69_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_69_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_69_OUT_SPK_SCALE,
		FC_LIF_LAYER_69_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_69;
}


NNLayer* Init_fc_lif_layer_70() {

	NNLayer* fc_lif_layer_70 = FC_LIF_Layer_Init(
		Getfc_lif_layer_70CMSPointer(),
		Getfc_lif_layer_70CMSLen(),
		Getfc_lif_layer_70WeightsPointer(),
		Getfc_lif_layer_70WeightsLen(),
		Getfc_lif_layer_70LIFParamPointer(),
		Getfc_lif_layer_70LIFParamLen(),
		Getfc_lif_layer_70LUTPointer(),
		Getfc_lif_layer_70LUTLen(),
		FC_LIF_LAYER_70_IS_LAST_LAYER,
		FC_LIF_LAYER_70_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_70_OUT_SPK_SCALE,
		FC_LIF_LAYER_70_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_70_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_70_TENSOR_ARENA_SIZE,
		fc_lif_layer_70_tensor_arena,
		fc_lif_layer_69_out_spk,
		fc_lif_layer_70_out_spk,
		FC_LIF_LAYER_70_BIAS_ADDR,
		FC_LIF_LAYER_70_WEIGHT_ADDR,
		FC_LIF_LAYER_70_V_MEM_ADDR,
		FC_LIF_LAYER_70_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_70_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_70_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_70_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_70_BIAS_LEN,
		FC_LIF_LAYER_70_WEIGHT_LEN,
		FC_LIF_LAYER_70_IN_SPK_SCALE,
		FC_LIF_LAYER_70_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_70_V_MEM_SCALE,
		FC_LIF_LAYER_70_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_70_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_70_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_70_OUT_SPK_SCALE,
		FC_LIF_LAYER_70_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_70;
}


NNLayer* Init_fc_lif_layer_71() {

	NNLayer* fc_lif_layer_71 = FC_LIF_Layer_Init(
		Getfc_lif_layer_71CMSPointer(),
		Getfc_lif_layer_71CMSLen(),
		Getfc_lif_layer_71WeightsPointer(),
		Getfc_lif_layer_71WeightsLen(),
		Getfc_lif_layer_71LIFParamPointer(),
		Getfc_lif_layer_71LIFParamLen(),
		Getfc_lif_layer_71LUTPointer(),
		Getfc_lif_layer_71LUTLen(),
		FC_LIF_LAYER_71_IS_LAST_LAYER,
		FC_LIF_LAYER_71_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_71_OUT_SPK_SCALE,
		FC_LIF_LAYER_71_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_71_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_71_TENSOR_ARENA_SIZE,
		fc_lif_layer_71_tensor_arena,
		fc_lif_layer_70_out_spk,
		fc_lif_layer_71_out_spk,
		FC_LIF_LAYER_71_BIAS_ADDR,
		FC_LIF_LAYER_71_WEIGHT_ADDR,
		FC_LIF_LAYER_71_V_MEM_ADDR,
		FC_LIF_LAYER_71_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_71_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_71_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_71_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_71_BIAS_LEN,
		FC_LIF_LAYER_71_WEIGHT_LEN,
		FC_LIF_LAYER_71_IN_SPK_SCALE,
		FC_LIF_LAYER_71_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_71_V_MEM_SCALE,
		FC_LIF_LAYER_71_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_71_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_71_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_71_OUT_SPK_SCALE,
		FC_LIF_LAYER_71_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_71;
}


NNLayer* Init_fc_lif_layer_72() {

	NNLayer* fc_lif_layer_72 = FC_LIF_Layer_Init(
		Getfc_lif_layer_72CMSPointer(),
		Getfc_lif_layer_72CMSLen(),
		Getfc_lif_layer_72WeightsPointer(),
		Getfc_lif_layer_72WeightsLen(),
		Getfc_lif_layer_72LIFParamPointer(),
		Getfc_lif_layer_72LIFParamLen(),
		Getfc_lif_layer_72LUTPointer(),
		Getfc_lif_layer_72LUTLen(),
		FC_LIF_LAYER_72_IS_LAST_LAYER,
		FC_LIF_LAYER_72_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_72_OUT_SPK_SCALE,
		FC_LIF_LAYER_72_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_72_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_72_TENSOR_ARENA_SIZE,
		fc_lif_layer_72_tensor_arena,
		fc_lif_layer_71_out_spk,
		fc_lif_layer_72_out_spk,
		FC_LIF_LAYER_72_BIAS_ADDR,
		FC_LIF_LAYER_72_WEIGHT_ADDR,
		FC_LIF_LAYER_72_V_MEM_ADDR,
		FC_LIF_LAYER_72_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_72_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_72_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_72_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_72_BIAS_LEN,
		FC_LIF_LAYER_72_WEIGHT_LEN,
		FC_LIF_LAYER_72_IN_SPK_SCALE,
		FC_LIF_LAYER_72_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_72_V_MEM_SCALE,
		FC_LIF_LAYER_72_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_72_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_72_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_72_OUT_SPK_SCALE,
		FC_LIF_LAYER_72_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_72;
}


NNLayer* Init_fc_lif_layer_73() {

	NNLayer* fc_lif_layer_73 = FC_LIF_Layer_Init(
		Getfc_lif_layer_73CMSPointer(),
		Getfc_lif_layer_73CMSLen(),
		Getfc_lif_layer_73WeightsPointer(),
		Getfc_lif_layer_73WeightsLen(),
		Getfc_lif_layer_73LIFParamPointer(),
		Getfc_lif_layer_73LIFParamLen(),
		Getfc_lif_layer_73LUTPointer(),
		Getfc_lif_layer_73LUTLen(),
		FC_LIF_LAYER_73_IS_LAST_LAYER,
		FC_LIF_LAYER_73_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_73_OUT_SPK_SCALE,
		FC_LIF_LAYER_73_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_73_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_73_TENSOR_ARENA_SIZE,
		fc_lif_layer_73_tensor_arena,
		fc_lif_layer_72_out_spk,
		fc_lif_layer_73_out_spk,
		FC_LIF_LAYER_73_BIAS_ADDR,
		FC_LIF_LAYER_73_WEIGHT_ADDR,
		FC_LIF_LAYER_73_V_MEM_ADDR,
		FC_LIF_LAYER_73_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_73_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_73_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_73_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_73_BIAS_LEN,
		FC_LIF_LAYER_73_WEIGHT_LEN,
		FC_LIF_LAYER_73_IN_SPK_SCALE,
		FC_LIF_LAYER_73_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_73_V_MEM_SCALE,
		FC_LIF_LAYER_73_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_73_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_73_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_73_OUT_SPK_SCALE,
		FC_LIF_LAYER_73_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_73;
}


NNLayer* Init_fc_lif_layer_74() {

	NNLayer* fc_lif_layer_74 = FC_LIF_Layer_Init(
		Getfc_lif_layer_74CMSPointer(),
		Getfc_lif_layer_74CMSLen(),
		Getfc_lif_layer_74WeightsPointer(),
		Getfc_lif_layer_74WeightsLen(),
		Getfc_lif_layer_74LIFParamPointer(),
		Getfc_lif_layer_74LIFParamLen(),
		Getfc_lif_layer_74LUTPointer(),
		Getfc_lif_layer_74LUTLen(),
		FC_LIF_LAYER_74_IS_LAST_LAYER,
		FC_LIF_LAYER_74_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_74_OUT_SPK_SCALE,
		FC_LIF_LAYER_74_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_74_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_74_TENSOR_ARENA_SIZE,
		fc_lif_layer_74_tensor_arena,
		fc_lif_layer_73_out_spk,
		fc_lif_layer_74_out_spk,
		FC_LIF_LAYER_74_BIAS_ADDR,
		FC_LIF_LAYER_74_WEIGHT_ADDR,
		FC_LIF_LAYER_74_V_MEM_ADDR,
		FC_LIF_LAYER_74_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_74_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_74_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_74_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_74_BIAS_LEN,
		FC_LIF_LAYER_74_WEIGHT_LEN,
		FC_LIF_LAYER_74_IN_SPK_SCALE,
		FC_LIF_LAYER_74_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_74_V_MEM_SCALE,
		FC_LIF_LAYER_74_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_74_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_74_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_74_OUT_SPK_SCALE,
		FC_LIF_LAYER_74_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_74;
}


NNLayer* Init_fc_lif_layer_75() {

	NNLayer* fc_lif_layer_75 = FC_LIF_Layer_Init(
		Getfc_lif_layer_75CMSPointer(),
		Getfc_lif_layer_75CMSLen(),
		Getfc_lif_layer_75WeightsPointer(),
		Getfc_lif_layer_75WeightsLen(),
		Getfc_lif_layer_75LIFParamPointer(),
		Getfc_lif_layer_75LIFParamLen(),
		Getfc_lif_layer_75LUTPointer(),
		Getfc_lif_layer_75LUTLen(),
		FC_LIF_LAYER_75_IS_LAST_LAYER,
		FC_LIF_LAYER_75_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_75_OUT_SPK_SCALE,
		FC_LIF_LAYER_75_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_75_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_75_TENSOR_ARENA_SIZE,
		fc_lif_layer_75_tensor_arena,
		fc_lif_layer_74_out_spk,
		fc_lif_layer_75_out_spk,
		FC_LIF_LAYER_75_BIAS_ADDR,
		FC_LIF_LAYER_75_WEIGHT_ADDR,
		FC_LIF_LAYER_75_V_MEM_ADDR,
		FC_LIF_LAYER_75_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_75_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_75_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_75_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_75_BIAS_LEN,
		FC_LIF_LAYER_75_WEIGHT_LEN,
		FC_LIF_LAYER_75_IN_SPK_SCALE,
		FC_LIF_LAYER_75_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_75_V_MEM_SCALE,
		FC_LIF_LAYER_75_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_75_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_75_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_75_OUT_SPK_SCALE,
		FC_LIF_LAYER_75_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_75;
}


NNLayer* Init_fc_lif_layer_76() {

	NNLayer* fc_lif_layer_76 = FC_LIF_Layer_Init(
		Getfc_lif_layer_76CMSPointer(),
		Getfc_lif_layer_76CMSLen(),
		Getfc_lif_layer_76WeightsPointer(),
		Getfc_lif_layer_76WeightsLen(),
		Getfc_lif_layer_76LIFParamPointer(),
		Getfc_lif_layer_76LIFParamLen(),
		Getfc_lif_layer_76LUTPointer(),
		Getfc_lif_layer_76LUTLen(),
		FC_LIF_LAYER_76_IS_LAST_LAYER,
		FC_LIF_LAYER_76_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_76_OUT_SPK_SCALE,
		FC_LIF_LAYER_76_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_76_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_76_TENSOR_ARENA_SIZE,
		fc_lif_layer_76_tensor_arena,
		fc_lif_layer_75_out_spk,
		fc_lif_layer_76_out_spk,
		FC_LIF_LAYER_76_BIAS_ADDR,
		FC_LIF_LAYER_76_WEIGHT_ADDR,
		FC_LIF_LAYER_76_V_MEM_ADDR,
		FC_LIF_LAYER_76_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_76_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_76_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_76_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_76_BIAS_LEN,
		FC_LIF_LAYER_76_WEIGHT_LEN,
		FC_LIF_LAYER_76_IN_SPK_SCALE,
		FC_LIF_LAYER_76_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_76_V_MEM_SCALE,
		FC_LIF_LAYER_76_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_76_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_76_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_76_OUT_SPK_SCALE,
		FC_LIF_LAYER_76_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_76;
}


NNLayer* Init_fc_lif_layer_77() {

	NNLayer* fc_lif_layer_77 = FC_LIF_Layer_Init(
		Getfc_lif_layer_77CMSPointer(),
		Getfc_lif_layer_77CMSLen(),
		Getfc_lif_layer_77WeightsPointer(),
		Getfc_lif_layer_77WeightsLen(),
		Getfc_lif_layer_77LIFParamPointer(),
		Getfc_lif_layer_77LIFParamLen(),
		Getfc_lif_layer_77LUTPointer(),
		Getfc_lif_layer_77LUTLen(),
		FC_LIF_LAYER_77_IS_LAST_LAYER,
		FC_LIF_LAYER_77_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_77_OUT_SPK_SCALE,
		FC_LIF_LAYER_77_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_77_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_77_TENSOR_ARENA_SIZE,
		fc_lif_layer_77_tensor_arena,
		fc_lif_layer_76_out_spk,
		fc_lif_layer_77_out_spk,
		FC_LIF_LAYER_77_BIAS_ADDR,
		FC_LIF_LAYER_77_WEIGHT_ADDR,
		FC_LIF_LAYER_77_V_MEM_ADDR,
		FC_LIF_LAYER_77_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_77_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_77_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_77_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_77_BIAS_LEN,
		FC_LIF_LAYER_77_WEIGHT_LEN,
		FC_LIF_LAYER_77_IN_SPK_SCALE,
		FC_LIF_LAYER_77_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_77_V_MEM_SCALE,
		FC_LIF_LAYER_77_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_77_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_77_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_77_OUT_SPK_SCALE,
		FC_LIF_LAYER_77_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_77;
}


NNLayer* Init_fc_lif_layer_78() {

	NNLayer* fc_lif_layer_78 = FC_LIF_Layer_Init(
		Getfc_lif_layer_78CMSPointer(),
		Getfc_lif_layer_78CMSLen(),
		Getfc_lif_layer_78WeightsPointer(),
		Getfc_lif_layer_78WeightsLen(),
		Getfc_lif_layer_78LIFParamPointer(),
		Getfc_lif_layer_78LIFParamLen(),
		Getfc_lif_layer_78LUTPointer(),
		Getfc_lif_layer_78LUTLen(),
		FC_LIF_LAYER_78_IS_LAST_LAYER,
		FC_LIF_LAYER_78_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_78_OUT_SPK_SCALE,
		FC_LIF_LAYER_78_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_78_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_78_TENSOR_ARENA_SIZE,
		fc_lif_layer_78_tensor_arena,
		fc_lif_layer_77_out_spk,
		fc_lif_layer_78_out_spk,
		FC_LIF_LAYER_78_BIAS_ADDR,
		FC_LIF_LAYER_78_WEIGHT_ADDR,
		FC_LIF_LAYER_78_V_MEM_ADDR,
		FC_LIF_LAYER_78_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_78_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_78_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_78_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_78_BIAS_LEN,
		FC_LIF_LAYER_78_WEIGHT_LEN,
		FC_LIF_LAYER_78_IN_SPK_SCALE,
		FC_LIF_LAYER_78_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_78_V_MEM_SCALE,
		FC_LIF_LAYER_78_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_78_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_78_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_78_OUT_SPK_SCALE,
		FC_LIF_LAYER_78_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_78;
}


NNLayer* Init_fc_lif_layer_79() {

	NNLayer* fc_lif_layer_79 = FC_LIF_Layer_Init(
		Getfc_lif_layer_79CMSPointer(),
		Getfc_lif_layer_79CMSLen(),
		Getfc_lif_layer_79WeightsPointer(),
		Getfc_lif_layer_79WeightsLen(),
		Getfc_lif_layer_79LIFParamPointer(),
		Getfc_lif_layer_79LIFParamLen(),
		Getfc_lif_layer_79LUTPointer(),
		Getfc_lif_layer_79LUTLen(),
		FC_LIF_LAYER_79_IS_LAST_LAYER,
		FC_LIF_LAYER_79_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_79_OUT_SPK_SCALE,
		FC_LIF_LAYER_79_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_79_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_79_TENSOR_ARENA_SIZE,
		fc_lif_layer_79_tensor_arena,
		fc_lif_layer_78_out_spk,
		fc_lif_layer_79_out_spk,
		FC_LIF_LAYER_79_BIAS_ADDR,
		FC_LIF_LAYER_79_WEIGHT_ADDR,
		FC_LIF_LAYER_79_V_MEM_ADDR,
		FC_LIF_LAYER_79_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_79_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_79_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_79_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_79_BIAS_LEN,
		FC_LIF_LAYER_79_WEIGHT_LEN,
		FC_LIF_LAYER_79_IN_SPK_SCALE,
		FC_LIF_LAYER_79_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_79_V_MEM_SCALE,
		FC_LIF_LAYER_79_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_79_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_79_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_79_OUT_SPK_SCALE,
		FC_LIF_LAYER_79_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_79;
}


NNLayer* Init_fc_lif_layer_80() {

	NNLayer* fc_lif_layer_80 = FC_LIF_Layer_Init(
		Getfc_lif_layer_80CMSPointer(),
		Getfc_lif_layer_80CMSLen(),
		Getfc_lif_layer_80WeightsPointer(),
		Getfc_lif_layer_80WeightsLen(),
		Getfc_lif_layer_80LIFParamPointer(),
		Getfc_lif_layer_80LIFParamLen(),
		Getfc_lif_layer_80LUTPointer(),
		Getfc_lif_layer_80LUTLen(),
		FC_LIF_LAYER_80_IS_LAST_LAYER,
		FC_LIF_LAYER_80_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_80_OUT_SPK_SCALE,
		FC_LIF_LAYER_80_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_80_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_80_TENSOR_ARENA_SIZE,
		fc_lif_layer_80_tensor_arena,
		fc_lif_layer_79_out_spk,
		fc_lif_layer_80_out_spk,
		FC_LIF_LAYER_80_BIAS_ADDR,
		FC_LIF_LAYER_80_WEIGHT_ADDR,
		FC_LIF_LAYER_80_V_MEM_ADDR,
		FC_LIF_LAYER_80_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_80_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_80_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_80_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_80_BIAS_LEN,
		FC_LIF_LAYER_80_WEIGHT_LEN,
		FC_LIF_LAYER_80_IN_SPK_SCALE,
		FC_LIF_LAYER_80_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_80_V_MEM_SCALE,
		FC_LIF_LAYER_80_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_80_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_80_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_80_OUT_SPK_SCALE,
		FC_LIF_LAYER_80_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_80;
}


NNLayer* Init_fc_lif_layer_81() {

	NNLayer* fc_lif_layer_81 = FC_LIF_Layer_Init(
		Getfc_lif_layer_81CMSPointer(),
		Getfc_lif_layer_81CMSLen(),
		Getfc_lif_layer_81WeightsPointer(),
		Getfc_lif_layer_81WeightsLen(),
		Getfc_lif_layer_81LIFParamPointer(),
		Getfc_lif_layer_81LIFParamLen(),
		Getfc_lif_layer_81LUTPointer(),
		Getfc_lif_layer_81LUTLen(),
		FC_LIF_LAYER_81_IS_LAST_LAYER,
		FC_LIF_LAYER_81_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_81_OUT_SPK_SCALE,
		FC_LIF_LAYER_81_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_81_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_81_TENSOR_ARENA_SIZE,
		fc_lif_layer_81_tensor_arena,
		fc_lif_layer_80_out_spk,
		fc_lif_layer_81_out_spk,
		FC_LIF_LAYER_81_BIAS_ADDR,
		FC_LIF_LAYER_81_WEIGHT_ADDR,
		FC_LIF_LAYER_81_V_MEM_ADDR,
		FC_LIF_LAYER_81_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_81_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_81_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_81_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_81_BIAS_LEN,
		FC_LIF_LAYER_81_WEIGHT_LEN,
		FC_LIF_LAYER_81_IN_SPK_SCALE,
		FC_LIF_LAYER_81_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_81_V_MEM_SCALE,
		FC_LIF_LAYER_81_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_81_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_81_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_81_OUT_SPK_SCALE,
		FC_LIF_LAYER_81_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_81;
}


NNLayer* Init_fc_lif_layer_82() {

	NNLayer* fc_lif_layer_82 = FC_LIF_Layer_Init(
		Getfc_lif_layer_82CMSPointer(),
		Getfc_lif_layer_82CMSLen(),
		Getfc_lif_layer_82WeightsPointer(),
		Getfc_lif_layer_82WeightsLen(),
		Getfc_lif_layer_82LIFParamPointer(),
		Getfc_lif_layer_82LIFParamLen(),
		Getfc_lif_layer_82LUTPointer(),
		Getfc_lif_layer_82LUTLen(),
		FC_LIF_LAYER_82_IS_LAST_LAYER,
		FC_LIF_LAYER_82_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_82_OUT_SPK_SCALE,
		FC_LIF_LAYER_82_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_82_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_82_TENSOR_ARENA_SIZE,
		fc_lif_layer_82_tensor_arena,
		fc_lif_layer_81_out_spk,
		fc_lif_layer_82_out_spk,
		FC_LIF_LAYER_82_BIAS_ADDR,
		FC_LIF_LAYER_82_WEIGHT_ADDR,
		FC_LIF_LAYER_82_V_MEM_ADDR,
		FC_LIF_LAYER_82_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_82_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_82_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_82_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_82_BIAS_LEN,
		FC_LIF_LAYER_82_WEIGHT_LEN,
		FC_LIF_LAYER_82_IN_SPK_SCALE,
		FC_LIF_LAYER_82_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_82_V_MEM_SCALE,
		FC_LIF_LAYER_82_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_82_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_82_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_82_OUT_SPK_SCALE,
		FC_LIF_LAYER_82_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_82;
}


NNLayer* Init_fc_lif_layer_83() {

	NNLayer* fc_lif_layer_83 = FC_LIF_Layer_Init(
		Getfc_lif_layer_83CMSPointer(),
		Getfc_lif_layer_83CMSLen(),
		Getfc_lif_layer_83WeightsPointer(),
		Getfc_lif_layer_83WeightsLen(),
		Getfc_lif_layer_83LIFParamPointer(),
		Getfc_lif_layer_83LIFParamLen(),
		Getfc_lif_layer_83LUTPointer(),
		Getfc_lif_layer_83LUTLen(),
		FC_LIF_LAYER_83_IS_LAST_LAYER,
		FC_LIF_LAYER_83_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_83_OUT_SPK_SCALE,
		FC_LIF_LAYER_83_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_83_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_83_TENSOR_ARENA_SIZE,
		fc_lif_layer_83_tensor_arena,
		fc_lif_layer_82_out_spk,
		fc_lif_layer_83_out_spk,
		FC_LIF_LAYER_83_BIAS_ADDR,
		FC_LIF_LAYER_83_WEIGHT_ADDR,
		FC_LIF_LAYER_83_V_MEM_ADDR,
		FC_LIF_LAYER_83_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_83_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_83_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_83_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_83_BIAS_LEN,
		FC_LIF_LAYER_83_WEIGHT_LEN,
		FC_LIF_LAYER_83_IN_SPK_SCALE,
		FC_LIF_LAYER_83_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_83_V_MEM_SCALE,
		FC_LIF_LAYER_83_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_83_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_83_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_83_OUT_SPK_SCALE,
		FC_LIF_LAYER_83_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_83;
}


NNLayer* Init_fc_lif_layer_84() {

	NNLayer* fc_lif_layer_84 = FC_LIF_Layer_Init(
		Getfc_lif_layer_84CMSPointer(),
		Getfc_lif_layer_84CMSLen(),
		Getfc_lif_layer_84WeightsPointer(),
		Getfc_lif_layer_84WeightsLen(),
		Getfc_lif_layer_84LIFParamPointer(),
		Getfc_lif_layer_84LIFParamLen(),
		Getfc_lif_layer_84LUTPointer(),
		Getfc_lif_layer_84LUTLen(),
		FC_LIF_LAYER_84_IS_LAST_LAYER,
		FC_LIF_LAYER_84_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_84_OUT_SPK_SCALE,
		FC_LIF_LAYER_84_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_84_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_84_TENSOR_ARENA_SIZE,
		fc_lif_layer_84_tensor_arena,
		fc_lif_layer_83_out_spk,
		fc_lif_layer_84_out_spk,
		FC_LIF_LAYER_84_BIAS_ADDR,
		FC_LIF_LAYER_84_WEIGHT_ADDR,
		FC_LIF_LAYER_84_V_MEM_ADDR,
		FC_LIF_LAYER_84_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_84_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_84_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_84_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_84_BIAS_LEN,
		FC_LIF_LAYER_84_WEIGHT_LEN,
		FC_LIF_LAYER_84_IN_SPK_SCALE,
		FC_LIF_LAYER_84_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_84_V_MEM_SCALE,
		FC_LIF_LAYER_84_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_84_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_84_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_84_OUT_SPK_SCALE,
		FC_LIF_LAYER_84_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_84;
}


NNLayer* Init_fc_lif_layer_85() {

	NNLayer* fc_lif_layer_85 = FC_LIF_Layer_Init(
		Getfc_lif_layer_85CMSPointer(),
		Getfc_lif_layer_85CMSLen(),
		Getfc_lif_layer_85WeightsPointer(),
		Getfc_lif_layer_85WeightsLen(),
		Getfc_lif_layer_85LIFParamPointer(),
		Getfc_lif_layer_85LIFParamLen(),
		Getfc_lif_layer_85LUTPointer(),
		Getfc_lif_layer_85LUTLen(),
		FC_LIF_LAYER_85_IS_LAST_LAYER,
		FC_LIF_LAYER_85_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_85_OUT_SPK_SCALE,
		FC_LIF_LAYER_85_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_85_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_85_TENSOR_ARENA_SIZE,
		fc_lif_layer_85_tensor_arena,
		fc_lif_layer_84_out_spk,
		fc_lif_layer_85_out_spk,
		FC_LIF_LAYER_85_BIAS_ADDR,
		FC_LIF_LAYER_85_WEIGHT_ADDR,
		FC_LIF_LAYER_85_V_MEM_ADDR,
		FC_LIF_LAYER_85_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_85_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_85_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_85_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_85_BIAS_LEN,
		FC_LIF_LAYER_85_WEIGHT_LEN,
		FC_LIF_LAYER_85_IN_SPK_SCALE,
		FC_LIF_LAYER_85_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_85_V_MEM_SCALE,
		FC_LIF_LAYER_85_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_85_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_85_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_85_OUT_SPK_SCALE,
		FC_LIF_LAYER_85_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_85;
}


NNLayer* Init_fc_lif_layer_86() {

	NNLayer* fc_lif_layer_86 = FC_LIF_Layer_Init(
		Getfc_lif_layer_86CMSPointer(),
		Getfc_lif_layer_86CMSLen(),
		Getfc_lif_layer_86WeightsPointer(),
		Getfc_lif_layer_86WeightsLen(),
		Getfc_lif_layer_86LIFParamPointer(),
		Getfc_lif_layer_86LIFParamLen(),
		Getfc_lif_layer_86LUTPointer(),
		Getfc_lif_layer_86LUTLen(),
		FC_LIF_LAYER_86_IS_LAST_LAYER,
		FC_LIF_LAYER_86_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_86_OUT_SPK_SCALE,
		FC_LIF_LAYER_86_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_86_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_86_TENSOR_ARENA_SIZE,
		fc_lif_layer_86_tensor_arena,
		fc_lif_layer_85_out_spk,
		fc_lif_layer_86_out_spk,
		FC_LIF_LAYER_86_BIAS_ADDR,
		FC_LIF_LAYER_86_WEIGHT_ADDR,
		FC_LIF_LAYER_86_V_MEM_ADDR,
		FC_LIF_LAYER_86_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_86_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_86_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_86_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_86_BIAS_LEN,
		FC_LIF_LAYER_86_WEIGHT_LEN,
		FC_LIF_LAYER_86_IN_SPK_SCALE,
		FC_LIF_LAYER_86_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_86_V_MEM_SCALE,
		FC_LIF_LAYER_86_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_86_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_86_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_86_OUT_SPK_SCALE,
		FC_LIF_LAYER_86_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_86;
}


NNLayer* Init_fc_lif_layer_87() {

	NNLayer* fc_lif_layer_87 = FC_LIF_Layer_Init(
		Getfc_lif_layer_87CMSPointer(),
		Getfc_lif_layer_87CMSLen(),
		Getfc_lif_layer_87WeightsPointer(),
		Getfc_lif_layer_87WeightsLen(),
		Getfc_lif_layer_87LIFParamPointer(),
		Getfc_lif_layer_87LIFParamLen(),
		Getfc_lif_layer_87LUTPointer(),
		Getfc_lif_layer_87LUTLen(),
		FC_LIF_LAYER_87_IS_LAST_LAYER,
		FC_LIF_LAYER_87_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_87_OUT_SPK_SCALE,
		FC_LIF_LAYER_87_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_87_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_87_TENSOR_ARENA_SIZE,
		fc_lif_layer_87_tensor_arena,
		fc_lif_layer_86_out_spk,
		fc_lif_layer_87_out_spk,
		FC_LIF_LAYER_87_BIAS_ADDR,
		FC_LIF_LAYER_87_WEIGHT_ADDR,
		FC_LIF_LAYER_87_V_MEM_ADDR,
		FC_LIF_LAYER_87_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_87_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_87_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_87_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_87_BIAS_LEN,
		FC_LIF_LAYER_87_WEIGHT_LEN,
		FC_LIF_LAYER_87_IN_SPK_SCALE,
		FC_LIF_LAYER_87_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_87_V_MEM_SCALE,
		FC_LIF_LAYER_87_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_87_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_87_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_87_OUT_SPK_SCALE,
		FC_LIF_LAYER_87_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_87;
}


NNLayer* Init_fc_lif_layer_88() {

	NNLayer* fc_lif_layer_88 = FC_LIF_Layer_Init(
		Getfc_lif_layer_88CMSPointer(),
		Getfc_lif_layer_88CMSLen(),
		Getfc_lif_layer_88WeightsPointer(),
		Getfc_lif_layer_88WeightsLen(),
		Getfc_lif_layer_88LIFParamPointer(),
		Getfc_lif_layer_88LIFParamLen(),
		Getfc_lif_layer_88LUTPointer(),
		Getfc_lif_layer_88LUTLen(),
		FC_LIF_LAYER_88_IS_LAST_LAYER,
		FC_LIF_LAYER_88_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_88_OUT_SPK_SCALE,
		FC_LIF_LAYER_88_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_88_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_88_TENSOR_ARENA_SIZE,
		fc_lif_layer_88_tensor_arena,
		fc_lif_layer_87_out_spk,
		fc_lif_layer_88_out_spk,
		FC_LIF_LAYER_88_BIAS_ADDR,
		FC_LIF_LAYER_88_WEIGHT_ADDR,
		FC_LIF_LAYER_88_V_MEM_ADDR,
		FC_LIF_LAYER_88_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_88_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_88_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_88_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_88_BIAS_LEN,
		FC_LIF_LAYER_88_WEIGHT_LEN,
		FC_LIF_LAYER_88_IN_SPK_SCALE,
		FC_LIF_LAYER_88_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_88_V_MEM_SCALE,
		FC_LIF_LAYER_88_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_88_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_88_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_88_OUT_SPK_SCALE,
		FC_LIF_LAYER_88_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_88;
}


NNLayer* Init_fc_lif_layer_89() {

	NNLayer* fc_lif_layer_89 = FC_LIF_Layer_Init(
		Getfc_lif_layer_89CMSPointer(),
		Getfc_lif_layer_89CMSLen(),
		Getfc_lif_layer_89WeightsPointer(),
		Getfc_lif_layer_89WeightsLen(),
		Getfc_lif_layer_89LIFParamPointer(),
		Getfc_lif_layer_89LIFParamLen(),
		Getfc_lif_layer_89LUTPointer(),
		Getfc_lif_layer_89LUTLen(),
		FC_LIF_LAYER_89_IS_LAST_LAYER,
		FC_LIF_LAYER_89_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_89_OUT_SPK_SCALE,
		FC_LIF_LAYER_89_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_89_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_89_TENSOR_ARENA_SIZE,
		fc_lif_layer_89_tensor_arena,
		fc_lif_layer_88_out_spk,
		fc_lif_layer_89_out_spk,
		FC_LIF_LAYER_89_BIAS_ADDR,
		FC_LIF_LAYER_89_WEIGHT_ADDR,
		FC_LIF_LAYER_89_V_MEM_ADDR,
		FC_LIF_LAYER_89_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_89_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_89_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_89_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_89_BIAS_LEN,
		FC_LIF_LAYER_89_WEIGHT_LEN,
		FC_LIF_LAYER_89_IN_SPK_SCALE,
		FC_LIF_LAYER_89_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_89_V_MEM_SCALE,
		FC_LIF_LAYER_89_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_89_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_89_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_89_OUT_SPK_SCALE,
		FC_LIF_LAYER_89_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_89;
}


NNLayer* Init_fc_lif_layer_90() {

	NNLayer* fc_lif_layer_90 = FC_LIF_Layer_Init(
		Getfc_lif_layer_90CMSPointer(),
		Getfc_lif_layer_90CMSLen(),
		Getfc_lif_layer_90WeightsPointer(),
		Getfc_lif_layer_90WeightsLen(),
		Getfc_lif_layer_90LIFParamPointer(),
		Getfc_lif_layer_90LIFParamLen(),
		Getfc_lif_layer_90LUTPointer(),
		Getfc_lif_layer_90LUTLen(),
		FC_LIF_LAYER_90_IS_LAST_LAYER,
		FC_LIF_LAYER_90_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_90_OUT_SPK_SCALE,
		FC_LIF_LAYER_90_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_90_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_90_TENSOR_ARENA_SIZE,
		fc_lif_layer_90_tensor_arena,
		fc_lif_layer_89_out_spk,
		fc_lif_layer_90_out_spk,
		FC_LIF_LAYER_90_BIAS_ADDR,
		FC_LIF_LAYER_90_WEIGHT_ADDR,
		FC_LIF_LAYER_90_V_MEM_ADDR,
		FC_LIF_LAYER_90_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_90_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_90_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_90_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_90_BIAS_LEN,
		FC_LIF_LAYER_90_WEIGHT_LEN,
		FC_LIF_LAYER_90_IN_SPK_SCALE,
		FC_LIF_LAYER_90_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_90_V_MEM_SCALE,
		FC_LIF_LAYER_90_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_90_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_90_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_90_OUT_SPK_SCALE,
		FC_LIF_LAYER_90_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_90;
}


NNLayer* Init_fc_lif_layer_91() {

	NNLayer* fc_lif_layer_91 = FC_LIF_Layer_Init(
		Getfc_lif_layer_91CMSPointer(),
		Getfc_lif_layer_91CMSLen(),
		Getfc_lif_layer_91WeightsPointer(),
		Getfc_lif_layer_91WeightsLen(),
		Getfc_lif_layer_91LIFParamPointer(),
		Getfc_lif_layer_91LIFParamLen(),
		Getfc_lif_layer_91LUTPointer(),
		Getfc_lif_layer_91LUTLen(),
		FC_LIF_LAYER_91_IS_LAST_LAYER,
		FC_LIF_LAYER_91_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_91_OUT_SPK_SCALE,
		FC_LIF_LAYER_91_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_91_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_91_TENSOR_ARENA_SIZE,
		fc_lif_layer_91_tensor_arena,
		fc_lif_layer_90_out_spk,
		fc_lif_layer_91_out_spk,
		FC_LIF_LAYER_91_BIAS_ADDR,
		FC_LIF_LAYER_91_WEIGHT_ADDR,
		FC_LIF_LAYER_91_V_MEM_ADDR,
		FC_LIF_LAYER_91_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_91_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_91_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_91_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_91_BIAS_LEN,
		FC_LIF_LAYER_91_WEIGHT_LEN,
		FC_LIF_LAYER_91_IN_SPK_SCALE,
		FC_LIF_LAYER_91_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_91_V_MEM_SCALE,
		FC_LIF_LAYER_91_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_91_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_91_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_91_OUT_SPK_SCALE,
		FC_LIF_LAYER_91_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_91;
}


NNLayer* Init_fc_lif_layer_92() {

	NNLayer* fc_lif_layer_92 = FC_LIF_Layer_Init(
		Getfc_lif_layer_92CMSPointer(),
		Getfc_lif_layer_92CMSLen(),
		Getfc_lif_layer_92WeightsPointer(),
		Getfc_lif_layer_92WeightsLen(),
		Getfc_lif_layer_92LIFParamPointer(),
		Getfc_lif_layer_92LIFParamLen(),
		Getfc_lif_layer_92LUTPointer(),
		Getfc_lif_layer_92LUTLen(),
		FC_LIF_LAYER_92_IS_LAST_LAYER,
		FC_LIF_LAYER_92_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_92_OUT_SPK_SCALE,
		FC_LIF_LAYER_92_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_92_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_92_TENSOR_ARENA_SIZE,
		fc_lif_layer_92_tensor_arena,
		fc_lif_layer_91_out_spk,
		fc_lif_layer_92_out_spk,
		FC_LIF_LAYER_92_BIAS_ADDR,
		FC_LIF_LAYER_92_WEIGHT_ADDR,
		FC_LIF_LAYER_92_V_MEM_ADDR,
		FC_LIF_LAYER_92_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_92_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_92_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_92_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_92_BIAS_LEN,
		FC_LIF_LAYER_92_WEIGHT_LEN,
		FC_LIF_LAYER_92_IN_SPK_SCALE,
		FC_LIF_LAYER_92_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_92_V_MEM_SCALE,
		FC_LIF_LAYER_92_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_92_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_92_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_92_OUT_SPK_SCALE,
		FC_LIF_LAYER_92_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_92;
}


NNLayer* Init_fc_lif_layer_93() {

	NNLayer* fc_lif_layer_93 = FC_LIF_Layer_Init(
		Getfc_lif_layer_93CMSPointer(),
		Getfc_lif_layer_93CMSLen(),
		Getfc_lif_layer_93WeightsPointer(),
		Getfc_lif_layer_93WeightsLen(),
		Getfc_lif_layer_93LIFParamPointer(),
		Getfc_lif_layer_93LIFParamLen(),
		Getfc_lif_layer_93LUTPointer(),
		Getfc_lif_layer_93LUTLen(),
		FC_LIF_LAYER_93_IS_LAST_LAYER,
		FC_LIF_LAYER_93_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_93_OUT_SPK_SCALE,
		FC_LIF_LAYER_93_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_93_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_93_TENSOR_ARENA_SIZE,
		fc_lif_layer_93_tensor_arena,
		fc_lif_layer_92_out_spk,
		fc_lif_layer_93_out_spk,
		FC_LIF_LAYER_93_BIAS_ADDR,
		FC_LIF_LAYER_93_WEIGHT_ADDR,
		FC_LIF_LAYER_93_V_MEM_ADDR,
		FC_LIF_LAYER_93_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_93_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_93_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_93_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_93_BIAS_LEN,
		FC_LIF_LAYER_93_WEIGHT_LEN,
		FC_LIF_LAYER_93_IN_SPK_SCALE,
		FC_LIF_LAYER_93_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_93_V_MEM_SCALE,
		FC_LIF_LAYER_93_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_93_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_93_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_93_OUT_SPK_SCALE,
		FC_LIF_LAYER_93_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_93;
}


NNLayer* Init_fc_lif_layer_94() {

	NNLayer* fc_lif_layer_94 = FC_LIF_Layer_Init(
		Getfc_lif_layer_94CMSPointer(),
		Getfc_lif_layer_94CMSLen(),
		Getfc_lif_layer_94WeightsPointer(),
		Getfc_lif_layer_94WeightsLen(),
		Getfc_lif_layer_94LIFParamPointer(),
		Getfc_lif_layer_94LIFParamLen(),
		Getfc_lif_layer_94LUTPointer(),
		Getfc_lif_layer_94LUTLen(),
		FC_LIF_LAYER_94_IS_LAST_LAYER,
		FC_LIF_LAYER_94_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_94_OUT_SPK_SCALE,
		FC_LIF_LAYER_94_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_94_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_94_TENSOR_ARENA_SIZE,
		fc_lif_layer_94_tensor_arena,
		fc_lif_layer_93_out_spk,
		fc_lif_layer_94_out_spk,
		FC_LIF_LAYER_94_BIAS_ADDR,
		FC_LIF_LAYER_94_WEIGHT_ADDR,
		FC_LIF_LAYER_94_V_MEM_ADDR,
		FC_LIF_LAYER_94_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_94_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_94_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_94_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_94_BIAS_LEN,
		FC_LIF_LAYER_94_WEIGHT_LEN,
		FC_LIF_LAYER_94_IN_SPK_SCALE,
		FC_LIF_LAYER_94_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_94_V_MEM_SCALE,
		FC_LIF_LAYER_94_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_94_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_94_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_94_OUT_SPK_SCALE,
		FC_LIF_LAYER_94_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_94;
}


NNLayer* Init_fc_lif_layer_95() {

	NNLayer* fc_lif_layer_95 = FC_LIF_Layer_Init(
		Getfc_lif_layer_95CMSPointer(),
		Getfc_lif_layer_95CMSLen(),
		Getfc_lif_layer_95WeightsPointer(),
		Getfc_lif_layer_95WeightsLen(),
		Getfc_lif_layer_95LIFParamPointer(),
		Getfc_lif_layer_95LIFParamLen(),
		Getfc_lif_layer_95LUTPointer(),
		Getfc_lif_layer_95LUTLen(),
		FC_LIF_LAYER_95_IS_LAST_LAYER,
		FC_LIF_LAYER_95_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_95_OUT_SPK_SCALE,
		FC_LIF_LAYER_95_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_95_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_95_TENSOR_ARENA_SIZE,
		fc_lif_layer_95_tensor_arena,
		fc_lif_layer_94_out_spk,
		fc_lif_layer_95_out_spk,
		FC_LIF_LAYER_95_BIAS_ADDR,
		FC_LIF_LAYER_95_WEIGHT_ADDR,
		FC_LIF_LAYER_95_V_MEM_ADDR,
		FC_LIF_LAYER_95_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_95_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_95_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_95_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_95_BIAS_LEN,
		FC_LIF_LAYER_95_WEIGHT_LEN,
		FC_LIF_LAYER_95_IN_SPK_SCALE,
		FC_LIF_LAYER_95_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_95_V_MEM_SCALE,
		FC_LIF_LAYER_95_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_95_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_95_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_95_OUT_SPK_SCALE,
		FC_LIF_LAYER_95_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_95;
}


NNLayer* Init_fc_lif_layer_96() {

	NNLayer* fc_lif_layer_96 = FC_LIF_Layer_Init(
		Getfc_lif_layer_96CMSPointer(),
		Getfc_lif_layer_96CMSLen(),
		Getfc_lif_layer_96WeightsPointer(),
		Getfc_lif_layer_96WeightsLen(),
		Getfc_lif_layer_96LIFParamPointer(),
		Getfc_lif_layer_96LIFParamLen(),
		Getfc_lif_layer_96LUTPointer(),
		Getfc_lif_layer_96LUTLen(),
		FC_LIF_LAYER_96_IS_LAST_LAYER,
		FC_LIF_LAYER_96_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_96_OUT_SPK_SCALE,
		FC_LIF_LAYER_96_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_96_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_96_TENSOR_ARENA_SIZE,
		fc_lif_layer_96_tensor_arena,
		fc_lif_layer_95_out_spk,
		fc_lif_layer_96_out_spk,
		FC_LIF_LAYER_96_BIAS_ADDR,
		FC_LIF_LAYER_96_WEIGHT_ADDR,
		FC_LIF_LAYER_96_V_MEM_ADDR,
		FC_LIF_LAYER_96_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_96_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_96_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_96_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_96_BIAS_LEN,
		FC_LIF_LAYER_96_WEIGHT_LEN,
		FC_LIF_LAYER_96_IN_SPK_SCALE,
		FC_LIF_LAYER_96_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_96_V_MEM_SCALE,
		FC_LIF_LAYER_96_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_96_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_96_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_96_OUT_SPK_SCALE,
		FC_LIF_LAYER_96_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_96;
}


NNLayer* Init_fc_lif_layer_97() {

	NNLayer* fc_lif_layer_97 = FC_LIF_Layer_Init(
		Getfc_lif_layer_97CMSPointer(),
		Getfc_lif_layer_97CMSLen(),
		Getfc_lif_layer_97WeightsPointer(),
		Getfc_lif_layer_97WeightsLen(),
		Getfc_lif_layer_97LIFParamPointer(),
		Getfc_lif_layer_97LIFParamLen(),
		Getfc_lif_layer_97LUTPointer(),
		Getfc_lif_layer_97LUTLen(),
		FC_LIF_LAYER_97_IS_LAST_LAYER,
		FC_LIF_LAYER_97_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_97_OUT_SPK_SCALE,
		FC_LIF_LAYER_97_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_97_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_97_TENSOR_ARENA_SIZE,
		fc_lif_layer_97_tensor_arena,
		fc_lif_layer_96_out_spk,
		fc_lif_layer_97_out_spk,
		FC_LIF_LAYER_97_BIAS_ADDR,
		FC_LIF_LAYER_97_WEIGHT_ADDR,
		FC_LIF_LAYER_97_V_MEM_ADDR,
		FC_LIF_LAYER_97_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_97_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_97_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_97_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_97_BIAS_LEN,
		FC_LIF_LAYER_97_WEIGHT_LEN,
		FC_LIF_LAYER_97_IN_SPK_SCALE,
		FC_LIF_LAYER_97_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_97_V_MEM_SCALE,
		FC_LIF_LAYER_97_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_97_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_97_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_97_OUT_SPK_SCALE,
		FC_LIF_LAYER_97_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_97;
}


NNLayer* Init_fc_lif_layer_98() {

	NNLayer* fc_lif_layer_98 = FC_LIF_Layer_Init(
		Getfc_lif_layer_98CMSPointer(),
		Getfc_lif_layer_98CMSLen(),
		Getfc_lif_layer_98WeightsPointer(),
		Getfc_lif_layer_98WeightsLen(),
		Getfc_lif_layer_98LIFParamPointer(),
		Getfc_lif_layer_98LIFParamLen(),
		Getfc_lif_layer_98LUTPointer(),
		Getfc_lif_layer_98LUTLen(),
		FC_LIF_LAYER_98_IS_LAST_LAYER,
		FC_LIF_LAYER_98_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_98_OUT_SPK_SCALE,
		FC_LIF_LAYER_98_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_98_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_98_TENSOR_ARENA_SIZE,
		fc_lif_layer_98_tensor_arena,
		fc_lif_layer_97_out_spk,
		fc_lif_layer_98_out_spk,
		FC_LIF_LAYER_98_BIAS_ADDR,
		FC_LIF_LAYER_98_WEIGHT_ADDR,
		FC_LIF_LAYER_98_V_MEM_ADDR,
		FC_LIF_LAYER_98_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_98_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_98_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_98_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_98_BIAS_LEN,
		FC_LIF_LAYER_98_WEIGHT_LEN,
		FC_LIF_LAYER_98_IN_SPK_SCALE,
		FC_LIF_LAYER_98_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_98_V_MEM_SCALE,
		FC_LIF_LAYER_98_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_98_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_98_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_98_OUT_SPK_SCALE,
		FC_LIF_LAYER_98_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_98;
}


NNLayer* Init_fc_lif_layer_99() {

	NNLayer* fc_lif_layer_99 = FC_LIF_Layer_Init(
		Getfc_lif_layer_99CMSPointer(),
		Getfc_lif_layer_99CMSLen(),
		Getfc_lif_layer_99WeightsPointer(),
		Getfc_lif_layer_99WeightsLen(),
		Getfc_lif_layer_99LIFParamPointer(),
		Getfc_lif_layer_99LIFParamLen(),
		Getfc_lif_layer_99LUTPointer(),
		Getfc_lif_layer_99LUTLen(),
		FC_LIF_LAYER_99_IS_LAST_LAYER,
		FC_LIF_LAYER_99_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_99_OUT_SPK_SCALE,
		FC_LIF_LAYER_99_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_99_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_99_TENSOR_ARENA_SIZE,
		fc_lif_layer_99_tensor_arena,
		fc_lif_layer_98_out_spk,
		fc_lif_layer_99_out_spk,
		FC_LIF_LAYER_99_BIAS_ADDR,
		FC_LIF_LAYER_99_WEIGHT_ADDR,
		FC_LIF_LAYER_99_V_MEM_ADDR,
		FC_LIF_LAYER_99_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_99_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_99_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_99_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_99_BIAS_LEN,
		FC_LIF_LAYER_99_WEIGHT_LEN,
		FC_LIF_LAYER_99_IN_SPK_SCALE,
		FC_LIF_LAYER_99_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_99_V_MEM_SCALE,
		FC_LIF_LAYER_99_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_99_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_99_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_99_OUT_SPK_SCALE,
		FC_LIF_LAYER_99_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_99;
}


NNLayer* Init_fc_lif_layer_100() {

	NNLayer* fc_lif_layer_100 = FC_LIF_Layer_Init(
		Getfc_lif_layer_100CMSPointer(),
		Getfc_lif_layer_100CMSLen(),
		Getfc_lif_layer_100WeightsPointer(),
		Getfc_lif_layer_100WeightsLen(),
		Getfc_lif_layer_100LIFParamPointer(),
		Getfc_lif_layer_100LIFParamLen(),
		Getfc_lif_layer_100LUTPointer(),
		Getfc_lif_layer_100LUTLen(),
		FC_LIF_LAYER_100_IS_LAST_LAYER,
		FC_LIF_LAYER_100_OUT_SPK_SUM_ADDR,
		FC_LIF_LAYER_100_OUT_SPK_SCALE,
		FC_LIF_LAYER_100_OUT_SPK_SUM_ZERO_POINT,
		FC_LIF_LAYER_100_MAX_NUM_TENSORS_TO_TRACK,
		FC_LIF_LAYER_100_TENSOR_ARENA_SIZE,
		fc_lif_layer_100_tensor_arena,
		fc_lif_layer_99_out_spk,
		fc_lif_layer_100_out_spk,
		FC_LIF_LAYER_100_BIAS_ADDR,
		FC_LIF_LAYER_100_WEIGHT_ADDR,
		FC_LIF_LAYER_100_V_MEM_ADDR,
		FC_LIF_LAYER_100_TIME_NOT_UPDATED_ADDR,
		FC_LIF_LAYER_100_UPDATE_NXT_LAYER_ADDR,
		FC_LIF_LAYER_100_INPUT_LAYER_SIZE,
		FC_LIF_LAYER_100_OUTPUT_LAYER_SIZE,
		FC_LIF_LAYER_100_BIAS_LEN,
		FC_LIF_LAYER_100_WEIGHT_LEN,
		FC_LIF_LAYER_100_IN_SPK_SCALE,
		FC_LIF_LAYER_100_IN_SPK_ZERO_POINT,
		FC_LIF_LAYER_100_V_MEM_SCALE,
		FC_LIF_LAYER_100_V_MEM_ZERO_POINT,
		FC_LIF_LAYER_100_TIME_NOT_UPDATED_SCALE,
		FC_LIF_LAYER_100_TIME_NOT_UPDATED_ZERO_POINT,
		FC_LIF_LAYER_100_OUT_SPK_SCALE,
		FC_LIF_LAYER_100_OUT_SPK_ZERO_POINT
	);	 return fc_lif_layer_100;
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
	Init_fc_lif_layer_51,
	Init_fc_lif_layer_52,
	Init_fc_lif_layer_53,
	Init_fc_lif_layer_54,
	Init_fc_lif_layer_55,
	Init_fc_lif_layer_56,
	Init_fc_lif_layer_57,
	Init_fc_lif_layer_58,
	Init_fc_lif_layer_59,
	Init_fc_lif_layer_60,
	Init_fc_lif_layer_61,
	Init_fc_lif_layer_62,
	Init_fc_lif_layer_63,
	Init_fc_lif_layer_64,
	Init_fc_lif_layer_65,
	Init_fc_lif_layer_66,
	Init_fc_lif_layer_67,
	Init_fc_lif_layer_68,
	Init_fc_lif_layer_69,
	Init_fc_lif_layer_70,
	Init_fc_lif_layer_71,
	Init_fc_lif_layer_72,
	Init_fc_lif_layer_73,
	Init_fc_lif_layer_74,
	Init_fc_lif_layer_75,
	Init_fc_lif_layer_76,
	Init_fc_lif_layer_77,
	Init_fc_lif_layer_78,
	Init_fc_lif_layer_79,
	Init_fc_lif_layer_80,
	Init_fc_lif_layer_81,
	Init_fc_lif_layer_82,
	Init_fc_lif_layer_83,
	Init_fc_lif_layer_84,
	Init_fc_lif_layer_85,
	Init_fc_lif_layer_86,
	Init_fc_lif_layer_87,
	Init_fc_lif_layer_88,
	Init_fc_lif_layer_89,
	Init_fc_lif_layer_90,
	Init_fc_lif_layer_91,
	Init_fc_lif_layer_92,
	Init_fc_lif_layer_93,
	Init_fc_lif_layer_94,
	Init_fc_lif_layer_95,
	Init_fc_lif_layer_96,
	Init_fc_lif_layer_97,
	Init_fc_lif_layer_98,
	Init_fc_lif_layer_99,
	Init_fc_lif_layer_100,
};