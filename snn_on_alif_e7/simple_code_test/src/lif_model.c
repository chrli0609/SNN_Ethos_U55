
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>



#include "include/lif_model.h"
#include "include/matmul.h"
#include "include/nn_ops.h"
#include "include/extra_funcs.h"


//#include "nn_ops/membrane_update_python.h"
#include "include/my_mem_u.h"








// Updated NNLayer structure to include tensor names
typedef struct {
    float scale;
    int zero_point;
} tuple;

typedef struct {
    int8_t** tensor_ptrs;   // Array of int8_t pointers
    size_t* tensor_sizes;   // Array of tensor sizes
    tuple* quant_params;    // Array of quantization parameter tuples
    char** tensor_names;    // Array of tensor names
    size_t num_tensors;     // Number of tensors
} NNLayer;

// Initialize a new NNLayer with specified capacity
NNLayer* NNLayer_Init(size_t num_tensors) {
    NNLayer* layer = (NNLayer*)malloc(sizeof(NNLayer));
    if (!layer) {
        return NULL;  // Memory allocation failed
    }
    
    // Allocate memory for tensor pointers array
    layer->tensor_ptrs = (int8_t**)malloc(num_tensors * sizeof(int8_t*));
    if (!layer->tensor_ptrs) {
        free(layer);
        return NULL;
    }
    
    // Allocate memory for tensor sizes array
    layer->tensor_sizes = (size_t*)malloc(num_tensors * sizeof(size_t));
    if (!layer->tensor_sizes) {
        free(layer->tensor_ptrs);
        free(layer);
        return NULL;
    }
    
    // Allocate memory for quantization parameters array
    layer->quant_params = (tuple*)malloc(num_tensors * sizeof(tuple));
    if (!layer->quant_params) {
        free(layer->tensor_sizes);
        free(layer->tensor_ptrs);
        free(layer);
        return NULL;
    }
    
    // Allocate memory for tensor names array
    layer->tensor_names = (char**)malloc(num_tensors * sizeof(char*));
    if (!layer->tensor_names) {
        free(layer->quant_params);
        free(layer->tensor_sizes);
        free(layer->tensor_ptrs);
        free(layer);
        return NULL;
    }
    
    // Initialize all pointers to NULL
    for (size_t i = 0; i < num_tensors; i++) {
        layer->tensor_ptrs[i] = NULL;
        layer->tensor_sizes[i] = 0;
        layer->quant_params[i].scale = 0.0f;
        layer->quant_params[i].zero_point = 0;
        layer->tensor_names[i] = NULL;
    }
    
    layer->num_tensors = num_tensors;
    
    return layer;
}

// Assign a tensor to a specific element in the NNLayer
int NNLayer_Assign(NNLayer* layer, size_t element, int8_t* tensor_ptr, size_t tensor_size, 
                   float scale, int zero_point, const char* tensor_name) {
    // Check if layer is valid
    if (!layer) {
        return -1;  // Invalid layer
    }
    
    // Check if element index is valid
    if (element >= layer->num_tensors) {
        return -2;  // Invalid element index
    }
    
    // Free previous tensor if it exists
    if (layer->tensor_ptrs[element] != NULL) {
        free(layer->tensor_ptrs[element]);
    }
    
    // Free previous name if it exists
    if (layer->tensor_names[element] != NULL) {
        free(layer->tensor_names[element]);
    }
    
    // Assign new tensor pointer
    layer->tensor_ptrs[element] = tensor_ptr;
    
    // Set tensor size
    layer->tensor_sizes[element] = tensor_size;
    
    // Set quantization parameters
    layer->quant_params[element].scale = scale;
    layer->quant_params[element].zero_point = zero_point;
    
    // Allocate and copy tensor name
    if (tensor_name) {
        layer->tensor_names[element] = strdup(tensor_name);
        if (!layer->tensor_names[element]) {
            return -3;  // Memory allocation failed for name
        }
    } else {
        // Provide default name if none is specified
        char default_name[32];
        snprintf(default_name, sizeof(default_name), "tensor_%zu", element);
        layer->tensor_names[element] = strdup(default_name);
        if (!layer->tensor_names[element]) {
            return -3;  // Memory allocation failed for name
        }
    }
    
    return 0;  // Success
}

// Free all memory associated with an NNLayer
void NNLayer_Free(NNLayer* layer) {
    if (!layer) {
        return;  // Nothing to free
    }
    
    // Free all tensor data and names
    for (size_t i = 0; i < layer->num_tensors; i++) {
        if (layer->tensor_ptrs[i] != NULL) {
            free(layer->tensor_ptrs[i]);
        }
        if (layer->tensor_names[i] != NULL) {
            free(layer->tensor_names[i]);
        }
    }
    
    // Free arrays
    free(layer->tensor_ptrs);
    free(layer->tensor_sizes);
    free(layer->quant_params);
    free(layer->tensor_names);
    
    // Free the layer structure itself
    free(layer);
}


// Function to dequantize and print all tensors in an NNLayer
void NNLayer_DequantizeAndPrint(const NNLayer* layer) {
    if (!layer) {
        printf("Error: NULL layer provided\n");
        return;
    }
    
    printf("NNLayer with %zu tensors:\n", layer->num_tensors);
    
    // Iterate through each tensor in the layer
    for (size_t tensor_idx = 0; tensor_idx < layer->num_tensors; tensor_idx++) {
        // Skip NULL tensors
        if (layer->tensor_ptrs[tensor_idx] == NULL) {
            printf("  Tensor %zu (%s): NULL\n", 
                  tensor_idx, 
                  layer->tensor_names[tensor_idx] ? layer->tensor_names[tensor_idx] : "unnamed");
            continue;
        }
        
        // Get tensor data and parameters
        int8_t* tensor_data = layer->tensor_ptrs[tensor_idx];
        size_t tensor_size = layer->tensor_sizes[tensor_idx];
        float scale = layer->quant_params[tensor_idx].scale;
        int zero_point = layer->quant_params[tensor_idx].zero_point;
        const char* tensor_name = layer->tensor_names[tensor_idx] ? 
                                  layer->tensor_names[tensor_idx] : "unnamed";
        
        // Allocate memory for dequantized values
        float* dequantized = (float*)malloc(tensor_size * sizeof(float));
        if (!dequantized) {
            printf("  Tensor %zu (%s): Memory allocation failed for dequantization\n", 
                  tensor_idx, tensor_name);
            continue;
        }
        
        // Dequantize the tensor
        dequantize_array_int8_to_float(tensor_data, dequantized, tensor_size, scale, zero_point);
        
        // Print tensor information and values
        printf("  Tensor %zu (%s): size=%zu, scale=%.6f, zero_point=%d\n", 
               tensor_idx, tensor_name, tensor_size, scale, zero_point);
        printf("    Quantized values: ");
        for (size_t i = 0; i < tensor_size && i < 10; i++) {
            printf("%d ", tensor_data[i]);
        }
        if (tensor_size > 10) printf("...");
        printf("\n");
        
        printf("    Dequantized values: ");
        for (size_t i = 0; i < tensor_size && i < 10; i++) {
            printf("%.4f ", dequantized[i]);
        }
        if (tensor_size > 10) printf("...");
        printf("\n");
        
        // Free dequantized memory
        free(dequantized);
    }
}







int my_mem_update(
    float* in_spk,

    float* ln_beta,
    float* vth,
    float* v_mem,
    float* time_not_updated,

    float* out_spk
) {


        size_t tensor_arena_size = MY_MEM_U_TENSOR_ARENA_SIZE;
        int8_t tensor_arena[tensor_arena_size] __attribute__((aligned(16)));



        // Allocate Tensor Arena
        // Initialize the allocator
        PersistentAllocator allocator;
        PersistentAllocator_Init(&allocator, tensor_arena, tensor_arena_size);


        // Manually set the relative addresses
        int8_t* in_spk_quant = PersistentAllocator_GetAbsPointer(&allocator, 
            MY_MEM_U_IN_SPK_ADDR);
        
        int8_t* bias_arena = PersistentAllocator_GetAbsPointer(&allocator, 
            MY_MEM_U_BIAS_ADDR);

        int8_t* weight_arena = PersistentAllocator_GetAbsPointer(&allocator, 
            MY_MEM_U_WEIGHT_ADDR);



        int8_t* ln_beta_quant = PersistentAllocator_GetAbsPointer(&allocator, 
            MY_MEM_U_LN_BETA_ADDR);
        int8_t* vth_quant = PersistentAllocator_GetAbsPointer(&allocator, 
            MY_MEM_U_VTH_ADDR);
        int8_t* v_mem_quant = PersistentAllocator_GetAbsPointer(&allocator, 
            MY_MEM_U_V_MEM_ADDR);
        int8_t* time_not_updated_quant = PersistentAllocator_GetAbsPointer(&allocator, 
            MY_MEM_U_TIME_NOT_UPDATED_ADDR);


        int8_t* decay_quant = PersistentAllocator_GetAbsPointer(&allocator, 
            MY_MEM_U_DECAY_ADDR);

        int8_t* in_curr_quant = PersistentAllocator_GetAbsPointer(&allocator, 
            MY_MEM_U_IN_CURR_ADDR);

        int8_t* decayed_mem_quant = PersistentAllocator_GetAbsPointer(&allocator, 
                MY_MEM_U_DECAYED_MEM_ADDR);


        int8_t* update_nxt_layer_quant = PersistentAllocator_GetAbsPointer(&allocator, 
                MY_MEM_U_UPDATE_NXT_LAYER_ADDR);
        int8_t* out_spk_quant = PersistentAllocator_GetAbsPointer(&allocator, 
                MY_MEM_U_OUT_SPK_ADDR);

        
        


        // Quantize
        quantize_array_float_to_int8(in_spk, in_spk_quant, MY_MEM_U_INPUT_LAYER_SIZE, MY_MEM_U_IN_SPK_SCALE, MY_MEM_U_IN_SPK_ZERO_POINT);
        quantize_array_float_to_int8(ln_beta, ln_beta_quant, MY_MEM_U_OUTPUT_LAYER_SIZE,MY_MEM_U_LN_BETA_SCALE, MY_MEM_U_LN_BETA_ZERO_POINT);
        quantize_array_float_to_int8(vth, vth_quant, MY_MEM_U_OUTPUT_LAYER_SIZE,MY_MEM_U_VTH_SCALE, MY_MEM_U_VTH_ZERO_POINT);
        quantize_array_float_to_int8(v_mem, v_mem_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_V_MEM_SCALE, MY_MEM_U_V_MEM_ZERO_POINT);
        quantize_array_float_to_int8(time_not_updated, time_not_updated_quant, MY_MEM_U_OUTPUT_LAYER_SIZE,MY_MEM_U_TIME_NOT_UPDATED_SCALE, MY_MEM_U_TIME_NOT_UPDATED_ZERO_POINT);




        // Inject my integer values here!!!
        //for (size_t i = 0; i < MY_MEM_U_INPUT_LAYER_SIZE; i++){
        //    in_spk_quant[i] = 1;
        //}
        


        NNLayer* nnlayer = NNLayer_Init(11);
        NNLayer_Assign(nnlayer, 0, in_spk_quant, MY_MEM_U_INPUT_LAYER_SIZE, MY_MEM_U_IN_SPK_SCALE, MY_MEM_U_IN_SPK_ZERO_POINT, "in_spk_quant");

        NNLayer_Assign(nnlayer, 1, bias_arena, MY_MEM_U_BIAS_LEN, MY_MEM_U_WEIGHT_SCALE, MY_MEM_U_WEIGHT_ZERO_POINT, "bias_arena");
        NNLayer_Assign(nnlayer, 2, weight_arena, MY_MEM_U_WEIGHT_LEN, MY_MEM_U_WEIGHT_SCALE, MY_MEM_U_WEIGHT_ZERO_POINT, "weight_arena");

        NNLayer_Assign(nnlayer, 3, ln_beta_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_LN_BETA_SCALE, MY_MEM_U_LN_BETA_ZERO_POINT, "ln_beta_quant");
        NNLayer_Assign(nnlayer, 4, vth_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_VTH_SCALE, MY_MEM_U_VTH_ZERO_POINT, "vth_quant");
        NNLayer_Assign(nnlayer, 5, v_mem_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_V_MEM_SCALE, MY_MEM_U_V_MEM_ZERO_POINT, "v_mem_quant");
        NNLayer_Assign(nnlayer, 6, time_not_updated_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_TIME_NOT_UPDATED_SCALE, MY_MEM_U_TIME_NOT_UPDATED_ZERO_POINT, "time_not_updated_quant");

        
        // Tmp1
        NNLayer_Assign(nnlayer, 7, decay_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_DECAY_SCALE, MY_MEM_U_DECAY_ZERO_POINT, "decay_quant");
        // Tmp2
        NNLayer_Assign(nnlayer, 8, in_curr_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_IN_CURR_SCALE, MY_MEM_U_IN_CURR_ZERO_POINT, "in_curr_quant");
        
        // Output
        NNLayer_Assign(nnlayer, 9, update_nxt_layer_quant, 1, MY_MEM_U_UPDATE_NXT_LAYER_SCALE, MY_MEM_U_UPDATE_NXT_LAYER_ZERO_POINT, "update_nxt_layer_quant");
        NNLayer_Assign(nnlayer, 10, out_spk_quant, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_OUT_SPK_SCALE, MY_MEM_U_OUT_SPK_ZERO_POINT, "out_spk_quant");

        


        NNLayer_DequantizeAndPrint(nnlayer);


        // Free layer
        //NNLayer_Free(nnlayer);


        //size_t exp_lut_size = 256;
        //int8_t exp_lut [exp_lut_size];
        //for (size_t i; i < exp_lut_size; i++) {
            //exp_lut[i] = 26;
        //}
        


        // Set start and end for input tensors

        // Run NPU Membrane Update
        my_mem_u_npu(
            tensor_arena,
            tensor_arena_size,

            Getmy_mem_uCMSPointer(),
            Getmy_mem_uCMSLen(),
            Getmy_mem_uWeightsPointer(),
            Getmy_mem_uWeightsLen(),


            Getmy_mem_uLUTPointer(),
            Getmy_mem_uLUTLen()
        );



        //float in_curr [MY_MEM_U_OUTPUT_LAYER_SIZE];
        //float decayed_mem [MY_MEM_U_OUTPUT_LAYER_SIZE];

        // Dequant outputs
        //dequantize_array_int8_to_float(v_mem_quant, v_mem, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_V_MEM_SCALE, MY_MEM_U_V_MEM_ZERO_POINT);
        //dequantize_array_int8_to_float(in_curr_quant, in_curr, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_IN_CURR_SCALE, MY_MEM_U_IN_CURR_ZERO_POINT);
        //dequantize_array_int8_to_float(decayed_mem_quant, decayed_mem, MY_MEM_U_OUTPUT_LAYER_SIZE, MY_MEM_U_DECAYED_MEM_SCALE, MY_MEM_U_DECAYED_MEM_ZERO_POINT);

        // Print output
        //PrintFloatTensor("v_mem", v_mem, MY_MEM_U_OUTPUT_LAYER_SIZE);
        //PrintFloatTensor("out_spk", out_spk, MY_MEM_U_OUTPUT_LAYER_SIZE);
        //PrintFloatTensor("in_curr", in_curr, MY_MEM_U_OUTPUT_LAYER_SIZE);
        //PrintFloatTensor("decayed_mem", decayed_mem, MY_MEM_U_OUTPUT_LAYER_SIZE);



        NNLayer_DequantizeAndPrint(nnlayer);
        //NNLayer_Free(nnlayer);
}





// Inputs:
//  * v_mem, dim=(LAYER i)
//  * in_spk, dim=(layer i-1)
//  * weights, dim=(depends on encoding)
//  * beta, scalar
//  * threshold, scalar
// Output:
//  * v_mem, dim=(layer i)
//  * out_spk, dim=(layer i)
//int membrane_update(
//    size_t tensor_arena_size,
//    size_t input_layer_size,
//    size_t output_layer_size,
//
//    size_t num_time_steps_since_update,
//    size_t* beta_idx,
//
//
//    int8_t* in_spk,
//    int8_t* v_mem,
//
//    const int8_t* weight_tensor_ptr,
//
//    int8_t* out_spk
//) {
//
//
//    // LUT
//    float decay[output_layer_size];
//    for (size_t i = 0; i < output_layer_size; i++) {
//        decay[i] = LUT_decay[beta_idx[i]][num_time_steps_since_update];
//    }
//    //int8_t decay_quant = quantize_array_float_to_int8(decay, )
//
//
//
//    // Do computation on NPU
//    if (membrane_update_npu(
//        tensor_arena_size,
//        in_spk,
//        input_layer_size,
//        v_mem,
//        output_layer_size,
//        decay,
//        output_layer_size,
//
//        GetMemUpdatePythonCMSPointer(),
//        GetMemUpdatePythonCMSLen(),
//
//        GetMemUpdatePythonWeightsPointer(),
//        GetMemUpdatePythonWeightsLen(),
//
//        v_mem,
//        output_layer_size,
//        out_spk,
//        output_layer_size
//    ) != 0) {
//        printf("ERROR IN membrane_update_npu\n"); return -1;
//    }
//
//
//
//    
//
//    return 0;
//
//}


