#include "include/init_nn_model.h"


#include <stdio.h> //printf
#include <stdlib.h> //malloc



#include "include/extra_funcs.h" //quantize_array_float_to_int8()


// First Layer
#include "include/my_mem_u.h"





NN_Model* MLP_Init() {


    // 1. Allocate for Total Arena Tensor on Heap
    size_t total_arena_tensor_size = MY_MEM_U_TENSOR_ARENA_SIZE;
    int8_t* total_arena_tensor = (int8_t*)aligned_malloc(total_arena_tensor_size, MEM_ALIGNMENT);


    // Do this for each layer we have

    // First NNLayer
    size_t tensor_arena_size = MY_MEM_U_TENSOR_ARENA_SIZE;
    int8_t* tensor_arena = total_arena_tensor; 
    NNLayer* nnlayer = NNLayer_Init(tensor_arena, tensor_arena_size, 11);
    if (nnlayer == NULL) { printf("Error when initializing NN_layer\n"); }


    // Manually set the relative addresses
    int8_t* in_spk_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        MY_MEM_U_IN_SPK_ADDR);
        
    int8_t* bias_arena = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        MY_MEM_U_BIAS_ADDR);

    int8_t* weight_arena = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        MY_MEM_U_WEIGHT_ADDR);

    int8_t* ln_beta_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        MY_MEM_U_LN_BETA_ADDR);
    int8_t* vth_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        MY_MEM_U_VTH_ADDR);
    int8_t* v_mem_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        MY_MEM_U_V_MEM_ADDR);
    int8_t* time_not_updated_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        MY_MEM_U_TIME_NOT_UPDATED_ADDR);


    int8_t* tmp1_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        MY_MEM_U_DECAY_ADDR);

    int8_t* tmp2_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        MY_MEM_U_IN_CURR_ADDR);



    int8_t* update_nxt_layer_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
            MY_MEM_U_UPDATE_NXT_LAYER_ADDR);
    int8_t* out_spk_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
            MY_MEM_U_OUT_SPK_ADDR);



    // Store pointers to quantized tensors for the layer in a struct
    int out_code = NNLayer_Assign(nnlayer, 0, in_spk_quant, MY_MEM_U_INPUT_LAYER_SIZE, MY_MEM_U_IN_SPK_SCALE, MY_MEM_U_IN_SPK_ZERO_POINT, "in_spk_quant");

    NNLayer_Assign(nnlayer, 1, bias_arena, MY_MEM_U_BIAS_LEN, MY_MEM_U_WEIGHT_SCALE, MY_MEM_U_WEIGHT_ZERO_POINT, "bias_arena");
    NNLayer_Assign(nnlayer, 2, weight_arena, MY_MEM_U_WEIGHT_LEN, MY_MEM_U_WEIGHT_SCALE, MY_MEM_U_WEIGHT_ZERO_POINT, "weight_arena");

    NNLayer_Assign(nnlayer, 3, ln_beta_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_LN_BETA_SCALE, MY_MEM_U_LN_BETA_ZERO_POINT, "ln_beta_quant");
    NNLayer_Assign(nnlayer, 4, vth_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_VTH_SCALE, MY_MEM_U_VTH_ZERO_POINT, "vth_quant");
    NNLayer_Assign(nnlayer, 5, v_mem_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_V_MEM_SCALE, MY_MEM_U_V_MEM_ZERO_POINT, "v_mem_quant");
    NNLayer_Assign(nnlayer, 6, time_not_updated_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_TIME_NOT_UPDATED_SCALE, MY_MEM_U_TIME_NOT_UPDATED_ZERO_POINT, "time_not_updated_quant");

        
    // Tmp1 & Tmp2, no quantization params needed
    NNLayer_Assign(nnlayer, 7, tmp1_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, 1, 0, "tmp1_quant");
    NNLayer_Assign(nnlayer, 8, tmp2_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, 1, 0, "tmp2_quant");
        
    // Output
    NNLayer_Assign(nnlayer, 9, update_nxt_layer_quant, 1, MY_MEM_U_UPDATE_NXT_LAYER_SCALE, MY_MEM_U_UPDATE_NXT_LAYER_ZERO_POINT, "update_nxt_layer_quant");
    NNLayer_Assign(nnlayer, 10, out_spk_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_OUT_SPK_SCALE, MY_MEM_U_OUT_SPK_ZERO_POINT, "out_spk_quant");



    // 3. Create NN_Model
    NN_Model* mlp_model = NN_Model_Init(total_arena_tensor, nnlayer);


    return mlp_model;
}


int MLP_Quantize_Inputs(NN_Model* mlp_model, float* in_spk, float* ln_beta, float* vth, float* v_mem, float* time_not_updated) {

    NNLayer* nnlayer = mlp_model->first_nnlayer;

    // Quantize
    quantize_array_float_to_int8(in_spk, nnlayer->tensor_ptrs[0], MY_MEM_U_INPUT_LAYER_SIZE, MY_MEM_U_IN_SPK_SCALE, MY_MEM_U_IN_SPK_ZERO_POINT);
    quantize_array_float_to_int8(ln_beta, nnlayer->tensor_ptrs[3], MY_MEM_U_OUTPUT_LAYER_SIZE,MY_MEM_U_LN_BETA_SCALE, MY_MEM_U_LN_BETA_ZERO_POINT);
    quantize_array_float_to_int8(vth, nnlayer->tensor_ptrs[4], MY_MEM_U_OUTPUT_LAYER_SIZE,MY_MEM_U_VTH_SCALE, MY_MEM_U_VTH_ZERO_POINT);
    quantize_array_float_to_int8(v_mem, nnlayer->tensor_ptrs[5], MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_V_MEM_SCALE, MY_MEM_U_V_MEM_ZERO_POINT);
    quantize_array_float_to_int8(time_not_updated, nnlayer->tensor_ptrs[6], MY_MEM_U_OUTPUT_LAYER_SIZE,MY_MEM_U_TIME_NOT_UPDATED_SCALE, MY_MEM_U_TIME_NOT_UPDATED_ZERO_POINT);

    return 0;
}


int MLP_Free(NN_Model* mlp_model) {

    // Deallocate total tensor arena
    aligned_free(mlp_model->total_tensor_arena);
    mlp_model->total_tensor_arena = NULL;

    // Deallocate NNLayers
    NNLayer_Free(mlp_model->first_nnlayer);
    mlp_model->first_nnlayer = NULL;

    // Deallocate NN_Model
    free(mlp_model);
    mlp_model = NULL;


    return 0;

}