#pragma once

#include "../include/nn_data_structure.h"
#include "../model.h"


#define MODEL_NAME "784x64x64x10"


#include "layers/fc_lif_layer_0.h"
#include "layers/fc_lif_layer_1.h"
#include "layers/fc_lif_layer_2.h"



#define MLP_INPUT_LAYER_SIZE	FC_LIF_LAYER_0_INPUT_LAYER_SIZE
#define MLP_OUTPUT_LAYER_SIZE	FC_LIF_LAYER_2_OUTPUT_LAYER_SIZE

#define MLP_NUM_LAYERS 3
#define MLP_NUM_TIME_STEPS 25


// for test patterns
#include "test_patterns/pattern_4.h"

#define NUM_TEST_SAMPLES test_input_4_NUM_SAMPLES

volatile int8_t* get_test_target() {
	return test_target_4;
}
volatile int8_t (*get_test_inputs())[MLP_NUM_TIME_STEPS][MLP_INPUT_LAYER_SIZE] {
	return test_input_4;
}


NNLayer* Init_fc_lif_layer_0() {


int8_t* region_ptrs_fc_lif_layer_0 [7] =
{
	GetWEIGHTS_AND_BIASES_REGIONfc_lif_layer_0Pointer(),
	GetSRAM_SCRATCH_REGIONfc_lif_layer_0Pointer(),
	GetSRAM_FAST_SCRATCH_REGIONfc_lif_layer_0Pointer(),
	GetPARAMS_REGIONSfc_lif_layer_0Pointer(),
	GetLUT_REGIONfc_lif_layer_0Pointer(),
	GetINPUT_REGIONfc_lif_layer_0Pointer(),
	GetOUTPUT_REGIONfc_lif_layer_0Pointer(),
};

size_t region_sizes_fc_lif_layer_0 [7] =
{
	GetWEIGHTS_AND_BIASES_REGIONfc_lif_layer_0Len(),
	GetSRAM_SCRATCH_REGIONfc_lif_layer_0Len(),
	GetSRAM_FAST_SCRATCH_REGIONfc_lif_layer_0Len(),
	GetPARAMS_REGIONSfc_lif_layer_0Len(),
	GetLUT_REGIONfc_lif_layer_0Len(),
	GetINPUT_REGIONfc_lif_layer_0Len(),
	GetOUTPUT_REGIONfc_lif_layer_0Len(),
};

size_t region_numbers_fc_lif_layer_0 [7] =
{
	0,
	1,
	2,
	3,
	4,
	5,
	6,
};
	NNLayer* fc_lif_layer_0 = NNLayer_Init(
		15,
		7
	);

	NNLayer_Assign(
		fc_lif_layer_0,
		Getcmsfc_lif_layer_0Pointer(),
		Getcmsfc_lif_layer_0Len(),
		region_ptrs_fc_lif_layer_0,
		region_sizes_fc_lif_layer_0,
		region_numbers_fc_lif_layer_0,
		7,
		"IN_SPK",
		784,
		"OUT_SPK",
		64,
		Getnamefc_lif_layer_0Pointer(),
		Getrelative_addrfc_lif_layer_0Pointer(),
		Getregionfc_lif_layer_0Pointer(),
		Getsizefc_lif_layer_0Pointer(),
		Getscalefc_lif_layer_0Pointer(),
		Getzero_pointfc_lif_layer_0Pointer(),
		15,
		FC_LIF_LAYER_0_IS_LAST_LAYER
	);
	return fc_lif_layer_0;
}




NNLayer* Init_fc_lif_layer_1() {


int8_t* region_ptrs_fc_lif_layer_1 [7] =
{
	GetWEIGHTS_AND_BIASES_REGIONfc_lif_layer_1Pointer(),
	GetSRAM_SCRATCH_REGIONfc_lif_layer_1Pointer(),
	GetSRAM_FAST_SCRATCH_REGIONfc_lif_layer_1Pointer(),
	GetPARAMS_REGIONSfc_lif_layer_1Pointer(),
	GetLUT_REGIONfc_lif_layer_1Pointer(),
	GetOUTPUT_REGIONfc_lif_layer_0Pointer(),
	GetOUTPUT_REGIONfc_lif_layer_1Pointer(),
};

size_t region_sizes_fc_lif_layer_1 [7] =
{
	GetWEIGHTS_AND_BIASES_REGIONfc_lif_layer_1Len(),
	GetSRAM_SCRATCH_REGIONfc_lif_layer_1Len(),
	GetSRAM_FAST_SCRATCH_REGIONfc_lif_layer_1Len(),
	GetPARAMS_REGIONSfc_lif_layer_1Len(),
	GetLUT_REGIONfc_lif_layer_1Len(),
	GetOUTPUT_REGIONfc_lif_layer_0Len(),
	GetOUTPUT_REGIONfc_lif_layer_1Len(),
};

size_t region_numbers_fc_lif_layer_1 [7] =
{
	0,
	1,
	2,
	3,
	4,
	5,
	6,
};
	NNLayer* fc_lif_layer_1 = NNLayer_Init(
		15,
		7
	);

	NNLayer_Assign(
		fc_lif_layer_1,
		Getcmsfc_lif_layer_1Pointer(),
		Getcmsfc_lif_layer_1Len(),
		region_ptrs_fc_lif_layer_1,
		region_sizes_fc_lif_layer_1,
		region_numbers_fc_lif_layer_1,
		7,
		"IN_SPK",
		64,
		"OUT_SPK",
		64,
		Getnamefc_lif_layer_1Pointer(),
		Getrelative_addrfc_lif_layer_1Pointer(),
		Getregionfc_lif_layer_1Pointer(),
		Getsizefc_lif_layer_1Pointer(),
		Getscalefc_lif_layer_1Pointer(),
		Getzero_pointfc_lif_layer_1Pointer(),
		15,
		FC_LIF_LAYER_1_IS_LAST_LAYER
	);
	return fc_lif_layer_1;
}




NNLayer* Init_fc_lif_layer_2() {


int8_t* region_ptrs_fc_lif_layer_2 [7] =
{
	GetWEIGHTS_AND_BIASES_REGIONfc_lif_layer_2Pointer(),
	GetSRAM_SCRATCH_REGIONfc_lif_layer_2Pointer(),
	GetSRAM_FAST_SCRATCH_REGIONfc_lif_layer_2Pointer(),
	GetPARAMS_REGIONSfc_lif_layer_2Pointer(),
	GetLUT_REGIONfc_lif_layer_2Pointer(),
	GetOUTPUT_REGIONfc_lif_layer_1Pointer(),
	GetOUTPUT_REGIONfc_lif_layer_2Pointer(),
};

size_t region_sizes_fc_lif_layer_2 [7] =
{
	GetWEIGHTS_AND_BIASES_REGIONfc_lif_layer_2Len(),
	GetSRAM_SCRATCH_REGIONfc_lif_layer_2Len(),
	GetSRAM_FAST_SCRATCH_REGIONfc_lif_layer_2Len(),
	GetPARAMS_REGIONSfc_lif_layer_2Len(),
	GetLUT_REGIONfc_lif_layer_2Len(),
	GetOUTPUT_REGIONfc_lif_layer_1Len(),
	GetOUTPUT_REGIONfc_lif_layer_2Len(),
};

size_t region_numbers_fc_lif_layer_2 [7] =
{
	0,
	1,
	2,
	3,
	4,
	5,
	6,
};
	NNLayer* fc_lif_layer_2 = NNLayer_Init(
		16,
		7
	);

	NNLayer_Assign(
		fc_lif_layer_2,
		Getcmsfc_lif_layer_2Pointer(),
		Getcmsfc_lif_layer_2Len(),
		region_ptrs_fc_lif_layer_2,
		region_sizes_fc_lif_layer_2,
		region_numbers_fc_lif_layer_2,
		7,
		"IN_SPK",
		64,
		"OUT_SPK",
		16,
		Getnamefc_lif_layer_2Pointer(),
		Getrelative_addrfc_lif_layer_2Pointer(),
		Getregionfc_lif_layer_2Pointer(),
		Getsizefc_lif_layer_2Pointer(),
		Getscalefc_lif_layer_2Pointer(),
		Getzero_pointfc_lif_layer_2Pointer(),
		16,
		FC_LIF_LAYER_2_IS_LAST_LAYER
	);
	return fc_lif_layer_2;
}



NNLayer* (*init_layers_func[MLP_NUM_LAYERS]) (void) = {
	Init_fc_lif_layer_0,
	Init_fc_lif_layer_1,
	Init_fc_lif_layer_2,
};