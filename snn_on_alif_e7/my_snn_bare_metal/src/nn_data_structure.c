
#include "include/nn_data_structure.h"

#include <stdlib.h> //malloc/free
#include <stdio.h>  //printf


#include "include/extra_funcs.h"








// For allocating main tensor (all the tensor arenas)
void* aligned_malloc(size_t required_bytes, size_t alignment) {
    void* p1;       // Original pointer returned by malloc
    void** p2;      // Aligned pointer to return

    // We need extra space: enough to realign and to store the original pointer
    size_t offset = alignment - 1 + sizeof(void*);

    // Allocate extra memory
    p1 = malloc(required_bytes + offset);
    if (p1 == NULL) return NULL;

    // Align the pointer upward and store the original just before the aligned address
    p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;  // Store original pointer for later free()

    return p2;
}


void aligned_free(void* p) {
    free(((void**)p)[-1]);
}










// Aligns a pointer down to the nearest aligned address
void* AlignPointerDown(void* ptr, size_t alignment) {
    return (void*)((uintptr_t)ptr & ~(alignment - 1));
}

// Initializes the PersistentAllocator
void PersistentAllocator_Init(PersistentAllocator* allocator, int8_t* arena, size_t size) {
    allocator->buffer_head = arena;
    allocator->tail_temp = arena + size;
}

// Allocates a persistent buffer
void* PersistentAllocator_Allocate(PersistentAllocator* allocator, size_t size, size_t alignment) {
    int8_t* aligned_result = (int8_t*)AlignPointerDown(allocator->tail_temp - size, alignment);
    
    if (aligned_result < allocator->buffer_head) {
        printf("Memory allocation failed! Requested: %zu bytes\n", size);
        return NULL;
    }

    allocator->tail_temp = aligned_result;
    return aligned_result;
}


// Manually allocate relative addressing
void* PersistentAllocator_GetAbsPointer(PersistentAllocator* allocator, size_t relative_addr) {

    void* absolute_ptr = allocator->buffer_head + relative_addr;

    if (absolute_ptr != AlignPointerDown(absolute_ptr, MEM_ALIGNMENT)) {
        printf("Error: manually set pointer is not 16-bit aligned\n");
    }

    return absolute_ptr;
}




// Getters
int8_t* PersistentAllocator_GetBufferHead(PersistentAllocator* allocator) {
    return allocator->buffer_head;
}

int8_t* PersistentAllocator_GetTailTemp(PersistentAllocator* allocator) {
    return allocator->tail_temp;
}






// Initialize a new NNLayer with specified capacity
NNLayer* NNLayer_Init(int8_t* tensor_arena, size_t tensor_arena_size, size_t num_tensors) {
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


    // Assign tensor arena pointer
    layer->tensor_arena = tensor_arena;
    layer->tensor_arena_size = tensor_arena_size;

    // Initiate PersistentAllocator
    PersistentAllocator_Init(&layer->allocator, layer->tensor_arena, layer->tensor_arena_size);


    // Default Next Layer is NULL
    layer->next_layer = NULL;


    // Initialize Input and Output
    layer->input = NULL;
    layer->output = NULL;
    
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
    layer->quant_params[element].scale_reciprocal = 1/scale;
    
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
    

    
    // Free arrays
    free(layer->tensor_ptrs);
    free(layer->tensor_sizes);
    free(layer->quant_params);
    free(layer->tensor_names);


    
    // Free the layer structure itself
    free(layer);

    // Set assign it to NULL to make sure we dont try to access it later:
    layer = NULL;
}


//// Function to dequantize and print all tensors in an NNLayer
//void NNLayer_DequantizeAndPrint(const NNLayer* layer) {
    //if (!layer) {
        //printf("Error: NULL layer provided\n");
        //return;
    //}
    
    //printf("NNLayer with %zu tensors:\n", layer->num_tensors);
    
    //// Iterate through each tensor in the layer
    //for (size_t tensor_idx = 0; tensor_idx < layer->num_tensors; tensor_idx++) {
        //// Skip NULL tensors
        //if (layer->tensor_ptrs[tensor_idx] == NULL) {
            //printf("  Tensor %zu (%s): NULL\n", 
                  //tensor_idx, 
                  //layer->tensor_names[tensor_idx] ? layer->tensor_names[tensor_idx] : "unnamed");
            //continue;
        //}
        
        //// Get tensor data and parameters
        //int8_t* tensor_data = layer->tensor_ptrs[tensor_idx];
        //size_t tensor_size = layer->tensor_sizes[tensor_idx];
        //float scale = layer->quant_params[tensor_idx].scale;
        //int zero_point = layer->quant_params[tensor_idx].zero_point;
        //const char* tensor_name = layer->tensor_names[tensor_idx] ? 
                                  //layer->tensor_names[tensor_idx] : "unnamed";
        
        //// Allocate memory for dequantized values
        //float* dequantized = (float*)malloc(tensor_size * sizeof(float));
        //if (!dequantized) {
            //printf("ERROR:  Tensor %zu (%s): Memory allocation failed for dequantization\n", 
                  //tensor_idx, tensor_name);
            //continue;
        //}
        
        //// Dequantize the tensor
        //dequantize_array_int8_to_float(tensor_data, dequantized, tensor_size, scale, zero_point);
        
        //// Print tensor information and values
        //printf("  Tensor %zu (%s): size=%zu, scale=%.6f, zero_point=%d\n", 
               //tensor_idx, tensor_name, tensor_size, scale, zero_point);
        //printf("    Quantized values: ");
        //for (size_t i = 0; i < tensor_size && i < 10; i++) {
            //printf("%d ", tensor_data[i]);
        //}
        //if (tensor_size > 10) printf("...");
        //printf("\n");
        
        //printf("    Dequantized values: ");
        //for (size_t i = 0; i < tensor_size && i < 10; i++) {
            //printf("%.4f ", dequantized[i]);
        //}
        //if (tensor_size > 10) printf("...");
        //printf("\n");
        
        //// Free dequantized memory
        //free(dequantized);
    //}
//}

// Function to dequantize and print all tensors in an NNLayer
// No dynamic memory allocation - computes dequantized values on-the-fly
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
        
        // Print tensor information
        printf("  Tensor %zu (%s): size=%zu, scale=%.6f, zero_point=%d\n", 
               tensor_idx, tensor_name, tensor_size, scale, zero_point);
        
        // Print quantized values (first 10 or all if fewer)
        printf("    Quantized values: ");
        size_t print_count = tensor_size < 10 ? tensor_size : 10;
        for (size_t i = 0; i < print_count; i++) {
            printf("%d ", tensor_data[i]);
        }
        if (tensor_size > 10) printf("...");
        printf("\n");
        
        // Print dequantized values (computed on-the-fly)
        printf("    Dequantized values: ");
        for (size_t i = 0; i < print_count; i++) {
            // Inline dequantization: (quantized_value - zero_point) * scale
            float dequantized_value = (tensor_data[i] - zero_point) * scale;
            printf("%.4f ", dequantized_value);
        }
        if (tensor_size > 10) printf("...");
        printf("\n");
    }
}




// Assume that NNLayer structs are already linked together
NN_Model* NN_Model_Init(int8_t* total_arena_tensor, NNLayer* first_nnlayer) {
    // Init Total Tensor Arena and store its pointer
    NN_Model* nn_model = (NN_Model*)malloc(sizeof(NN_Model));
    
    
    //note: remove total_tensor_arena as attribute from NN_Model struct
    //nn_model->total_tensor_arena = total_arena_tensor;
    nn_model->first_nnlayer = first_nnlayer;

    // Set nn model input tensor
    nn_model->input = first_nnlayer->input;

    // Set nn model output tensor
    NNLayer* nnlayer = first_nnlayer;
    while (nnlayer->next_layer != NULL) {
        nnlayer = nnlayer->next_layer;
    }
    nn_model->output = nnlayer->output;





    return nn_model;


}
