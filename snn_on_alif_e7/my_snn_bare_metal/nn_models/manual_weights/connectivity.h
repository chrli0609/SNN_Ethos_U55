#pragma once

#include "../include/nn_data_structure.h"
#include "../model.h"


#define MODEL_NAME "manual_weights"


#include "layers/fc_lif_layer_0.h"
#include "layers/fc_lif_layer_1.h"



#define MLP_INPUT_LAYER_SIZE	FC_LIF_LAYER_0_INPUT_LAYER_SIZE
#define MLP_OUTPUT_LAYER_SIZE	FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE

#define MLP_NUM_LAYERS 2
#define MLP_NUM_TIME_STEPS 2


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
		16,
		"OUT_SPK",
		16,
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
		16,
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
		16,
		"OUT_SPK",
		16,
		Getnamefc_lif_layer_1Pointer(),
		Getrelative_addrfc_lif_layer_1Pointer(),
		Getregionfc_lif_layer_1Pointer(),
		Getsizefc_lif_layer_1Pointer(),
		Getscalefc_lif_layer_1Pointer(),
		Getzero_pointfc_lif_layer_1Pointer(),
		16,
		FC_LIF_LAYER_1_IS_LAST_LAYER
	);
	return fc_lif_layer_1;
}



NNLayer* (*init_layers_func[MLP_NUM_LAYERS]) (void) = {
	Init_fc_lif_layer_0,
	Init_fc_lif_layer_1,
};