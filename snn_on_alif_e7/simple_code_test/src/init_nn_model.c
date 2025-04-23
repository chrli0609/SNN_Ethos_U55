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

    size_t in_spk_relative_addr = MY_MEM_U_IN_SPK_ADDR;
    size_t bias_relative_addr = MY_MEM_U_BIAS_ADDR;
    size_t weight_relative_addr = MY_MEM_U_WEIGHT_ADDR;
    //size_t ln_beta_relative_addr = MY_MEM_U_LN_BETA_ADDR;
    //size_t vth_relative_addr = MY_MEM_U_VTH_ADDR;
    size_t v_mem_relative_addr = MY_MEM_U_V_MEM_ADDR;
    size_t time_not_updated_relative_addr = MY_MEM_U_TIME_NOT_UPDATED_ADDR;
    //size_t tmp1_relative_addr = MY_MEM_U_TMP1_ADDR;
    //size_t tmp2_relative_addr = MY_MEM_U_TMP2_ADDR;
    size_t update_nxt_layer_relative_addr = MY_MEM_U_UPDATE_NXT_LAYER_ADDR;
    size_t out_spk_relative_addr = MY_MEM_U_OUT_SPK_ADDR;
    



    size_t input_layer_size = MY_MEM_U_INPUT_LAYER_SIZE;
    size_t output_layer_size = MY_MEM_U_OUTPUT_LAYER_SIZE;
    size_t bias_tensor_size = MY_MEM_U_BIAS_LEN;
    size_t weight_tensor_size = MY_MEM_U_WEIGHT_LEN;


    float in_spk_scale = MY_MEM_U_IN_SPK_SCALE;
    int in_spk_zero_point = MY_MEM_U_IN_SPK_ZERO_POINT;

    //float ln_beta_scale = MY_MEM_U_LN_BETA_SCALE;
    //int ln_beta_zero_point = MY_MEM_U_LN_BETA_ZERO_POINT;
    //float vth_scale = MY_MEM_U_VTH_SCALE;
    //int vth_zero_point = MY_MEM_U_VTH_ZERO_POINT;
    float v_mem_scale = MY_MEM_U_V_MEM_SCALE;
    int v_mem_zero_point = MY_MEM_U_V_MEM_ZERO_POINT;
    float time_not_updated_scale = MY_MEM_U_TIME_NOT_UPDATED_SCALE;
    int time_not_updated_zero_point = MY_MEM_U_TIME_NOT_UPDATED_ZERO_POINT;

    float out_spk_scale = MY_MEM_U_OUT_SPK_SCALE;
    int out_spk_zero_point = MY_MEM_U_OUT_SPK_ZERO_POINT;



    NNLayer* nnlayer = NNLayer_Init(tensor_arena, tensor_arena_size, 7);
    if (nnlayer == NULL) { printf("Error when initializing NN_layer\n"); }


    // Manually set the relative addresses
    int8_t* in_spk_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        in_spk_relative_addr);
        
    int8_t* bias_arena = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        bias_relative_addr);

    int8_t* weight_arena = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        weight_relative_addr);
    //int8_t* ln_beta_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        //ln_beta_relative_addr);
    //int8_t* vth_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        //vth_relative_addr);
    int8_t* v_mem_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        v_mem_relative_addr);
    int8_t* time_not_updated_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        time_not_updated_relative_addr);
    //int8_t* tmp1_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        //MY_MEM_U_DECAY_ADDR);
    //int8_t* tmp2_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        //MY_MEM_U_IN_CURR_ADDR);
    int8_t* update_nxt_layer_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
            update_nxt_layer_relative_addr);
    int8_t* out_spk_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
            out_spk_relative_addr);



    // Store pointers to quantized tensors for the layer in a struct
    NNLayer_Assign(nnlayer, 0, in_spk_quant, input_layer_size, in_spk_scale, in_spk_zero_point, "in_spk_quant");

    NNLayer_Assign(nnlayer, 1, bias_arena, bias_tensor_size, 1, 0, "bias_arena");
    NNLayer_Assign(nnlayer, 2, weight_arena, weight_tensor_size, 1, 0, "weight_arena");

    //NNLayer_Assign(nnlayer, 3, ln_beta_quant, output_layer_size, ln_beta_scale, ln_beta_zero_point, "ln_beta_quant");
    //NNLayer_Assign(nnlayer, 4, vth_quant, output_layer_size, vth_scale, vth_zero_point, "vth_quant");
    NNLayer_Assign(nnlayer, 3, v_mem_quant, output_layer_size, v_mem_scale, v_mem_zero_point, "v_mem_quant");
    NNLayer_Assign(nnlayer, 4, time_not_updated_quant, output_layer_size, time_not_updated_scale, time_not_updated_zero_point, "time_not_updated_quant");

        
    // Tmp1 & Tmp2, no quantization params needed
    //NNLayer_Assign(nnlayer, 7, tmp1_quant, output_layer_size, 1, 0, "tmp1_quant");
    //NNLayer_Assign(nnlayer, 8, tmp2_quant, output_layer_size, 1, 0, "tmp2_quant");
        
    // Output
    NNLayer_Assign(nnlayer, 5, update_nxt_layer_quant, 1, 1, 0, "update_nxt_layer_quant");
    NNLayer_Assign(nnlayer, 6, out_spk_quant, output_layer_size, out_spk_scale, out_spk_zero_point, "out_spk_quant");



    // 3. Create NN_Model
    NN_Model* mlp_model = NN_Model_Init(total_arena_tensor, nnlayer);


    return mlp_model;
}


int MLP_Quantize_Inputs(NN_Model* mlp_model, float* in_spk, float* v_mem, float* time_not_updated) {

    NNLayer* nnlayer = mlp_model->first_nnlayer;

    // Quantize
    quantize_array_float_to_int8(in_spk, nnlayer->tensor_ptrs[0], MY_MEM_U_INPUT_LAYER_SIZE, MY_MEM_U_IN_SPK_SCALE, MY_MEM_U_IN_SPK_ZERO_POINT);
    //quantize_array_float_to_int8(ln_beta, nnlayer->tensor_ptrs[3], MY_MEM_U_OUTPUT_LAYER_SIZE,MY_MEM_U_LN_BETA_SCALE, MY_MEM_U_LN_BETA_ZERO_POINT);
    //quantize_array_float_to_int8(vth, nnlayer->tensor_ptrs[4], MY_MEM_U_OUTPUT_LAYER_SIZE,MY_MEM_U_VTH_SCALE, MY_MEM_U_VTH_ZERO_POINT);
    quantize_array_float_to_int8(v_mem, nnlayer->tensor_ptrs[3], MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_V_MEM_SCALE, MY_MEM_U_V_MEM_ZERO_POINT);
    quantize_array_float_to_int8(time_not_updated, nnlayer->tensor_ptrs[4], MY_MEM_U_OUTPUT_LAYER_SIZE,MY_MEM_U_TIME_NOT_UPDATED_SCALE, MY_MEM_U_TIME_NOT_UPDATED_ZERO_POINT);

    return 0;
}




#include "include/init_nn_model.h"
#include "include/nn_data_structure.h"
#include "include/nn_ops.h"

int MLP_Inference(
    NN_Model* mlp_model,

    float* in_spk,

    float* v_mem,
    float* time_not_updated,

    float* out_spk
) {

        MLP_Quantize_Inputs(mlp_model, in_spk, v_mem, time_not_updated);


        // First layer

        NNLayer* nnlayer = mlp_model->first_nnlayer;

        // Check Tensor Arena Values Before NPU OP
        NNLayer_DequantizeAndPrint(nnlayer);




        // Run NPU Membrane Update
        my_mem_u_npu(
            nnlayer->tensor_arena,
            nnlayer->tensor_arena_size,

            Getmy_mem_uCMSPointer(),
            Getmy_mem_uCMSLen(),
            Getmy_mem_uWeightsPointer(),
            Getmy_mem_uWeightsLen(),

            Getmy_mem_uLIFParamPointer(),
            Getmy_mem_uLIFParamLen(),
            Getmy_mem_uLUTPointer(),
            Getmy_mem_uLUTLen()
        );



        
        // Check resulting Tensor Arena Values after NPU OP
        NNLayer_DequantizeAndPrint(nnlayer);

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