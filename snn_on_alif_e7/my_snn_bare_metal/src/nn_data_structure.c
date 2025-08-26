#include "include/nn_data_structure.h"

#include <stdlib.h> //malloc/free
#include <stdio.h>  //printf

#include <string.h> //strcomp


#include "include/extra_funcs.h"
//#include "nn_models/spk_mnist_784x32x10/model.h"
//#include "nn_models/nmnist_784x64x64x10/model.h"
//#include "nn_models/nmnist_784x32x32x32x10/model.h"








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

//// Initializes the PersistentAllocator
//void PersistentAllocator_Init(PersistentAllocator* allocator, int8_t* arena, size_t size) {
    //allocator->buffer_head = arena;
    //allocator->tail_temp = arena + size;
//}

//// Allocates a persistent buffer
//void* PersistentAllocator_Allocate(PersistentAllocator* allocator, size_t size, size_t alignment) {
    //int8_t* aligned_result = (int8_t*)AlignPointerDown(allocator->tail_temp - size, alignment);
    
    //if (aligned_result < allocator->buffer_head) {
        //printf("Memory allocation failed! Requested: %zu bytes\n", size);
        //return NULL;
    //}

    //allocator->tail_temp = aligned_result;
    //return aligned_result;
//}


//// Manually allocate relative addressing

//void* PersistentAllocator_GetAbsPointer(PersistentAllocator* allocator, size_t relative_addr) {
    //void* absolute_ptr = allocator->buffer_head + relative_addr;

    //// Only an error if its the pointer to the whole tensor, not only pointers within the tensor
    ////if (absolute_ptr != AlignPointerDown(absolute_ptr, MEM_ALIGNMENT)) {
        //////printf("Error: manually set pointer is not 16-bit aligned\n");
    ////}

    //return absolute_ptr;
//}




//// Getters
//int8_t* PersistentAllocator_GetBufferHead(PersistentAllocator* allocator) {
    //return allocator->buffer_head;
//}

//int8_t* PersistentAllocator_GetTailTemp(PersistentAllocator* allocator) {
    //return allocator->tail_temp;
//}


int8_t* relative_addr_2_absolute_addr(int8_t* region_ptr, size_t relative_addr) {


    return region_ptr + relative_addr;
}


Tensor* Tensor_Init() {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));

    tensor->name = NULL;
    tensor->ptr = NULL;
    tensor->size = 0;
    tensor->region_number = 99;
    tensor->relative_addr = 999999;

    tensor->scale = 0;
    tensor->zero_point = 0;
    tensor->scale_reciprocal = 0;

    return tensor;
}

int Tensor_Assign(Tensor* tensor, const char* name, int8_t* region_ptr, size_t region_number, size_t relative_addr, size_t size, float scale, size_t zero_point) {
    tensor->name = name;
    tensor->relative_addr = relative_addr;
    tensor->region_number = region_number;
    tensor->size = size;

    tensor->ptr = relative_addr_2_absolute_addr(region_ptr, relative_addr);

    // Quantization params
    tensor->scale = scale;
    tensor->zero_point = zero_point;
    tensor->scale_reciprocal = 1 / scale;

    //Tensor_Print(tensor);

    return 0;

}

void Tensor_Print(Tensor* tensor) {
    printf("Tensor: %s\t\t(struct stored at: %p,\ttensor stored at: %p)\n", tensor->name, tensor, tensor->ptr);
    printf("\tRegion: %d\tRelative Address: %d\tSize: %d\n", tensor->region_number, tensor->relative_addr, tensor->size);
    printf("\tScale: %f\tZero Point: %d\n", tensor->scale, tensor->zero_point);
}


void Tensor_Print_Quant_Values(Tensor* tensor) {

    int8_t* arr = tensor->ptr;
    printf("Tensor: %s\n", tensor->name);
    for (size_t i = 0; i < tensor->size; i++) {
        printf("%d ", arr[i]);
        if (i % MAX_NUM_PRINTED_VALUES_PER_LINE == 0 && i != 0) { printf("\n"); }
    }
    printf("\n");
}

void Tensor_Print_Dequant_Values(Tensor* tensor) {

    int8_t* arr = tensor->ptr;
    printf("Tensor: %s\n", tensor->name);
    for (size_t i = 0; i < tensor->size; i++) {
        float dequantized_value = (arr[i] - tensor->zero_point) * tensor->scale;
        printf("%f ", dequantized_value);
        if (i % MAX_NUM_PRINTED_VALUES_PER_LINE == 0 && i != 0) { printf("\n"); }
    }
    printf("\n");
}


MemoryRegion* MemoryRegion_Init() {
    MemoryRegion* memory_region = (MemoryRegion*)malloc(sizeof(MemoryRegion));
    memory_region->region_start_ptr = NULL;
    memory_region->length = 0;

    return memory_region;
}


// Initialize a new NNLayer with specified capacity
NNLayer* NNLayer_Init(size_t num_tensors, size_t num_regions) {
    NNLayer* layer = (NNLayer*)malloc(sizeof(NNLayer));
    if (!layer) {
        return NULL;  // Memory allocation failed
    }
    

    layer->tensors = (Tensor**)malloc(num_tensors * sizeof(Tensor*));
    if (!layer->tensors) {
        free(layer);
    }

    // Allocate memory for array containing custom memory regions
    layer->memory_regions = (MemoryRegion**)malloc(num_regions * sizeof(MemoryRegion*));
    if (!layer->memory_regions) {
        //free(layer->quant_params);
        //free(layer->tensor_sizes);
        //free(layer->tensor_ptrs);
        //free(layer->tensor_names);
        free(layer->tensors);
        free(layer);
        return NULL;
    }



    // Initiate PersistentAllocator for each region
    //PersistentAllocator_Init(&layer->allocator, layer->sram_scratch_region->region_start_ptr, layer->sram_scratch_region->length);

    //for (size_t i = 0; i < num_regions; i++) {
        //PersistentAllocator_Init(&layer->allocator, layer->custom_regions[i]->region_start_ptr, layer->custom_regions[i]->length);

    //}


    // Default Next Layer is NULL
    layer->next_layer = NULL;


    // Initialize Tensors
    layer->input = Tensor_Init();
    layer->output = Tensor_Init();
    layer->update_nxt_layer = Tensor_Init();
    layer->update_curr_layer = Tensor_Init();


    // Initialize Tensors
    layer->num_tensors = num_tensors;
    for (size_t i = 0; i < num_tensors; i++) {
        layer->tensors[i] = Tensor_Init();
    }

    // Initialize Memory Regions
    layer->num_regions = num_regions;
    for (size_t i = 0; i < num_regions; i++) {
        layer->memory_regions[i] = MemoryRegion_Init();
    }

    

    return layer;
}




int NNLayer_Assign(
    NNLayer* nnlayer,
    const uint8_t* command_stream,
    size_t command_stream_length,

    int8_t** memory_region_ptrs,
    size_t* memory_region_sizes,
    size_t* memory_region_region_numbers,
    size_t num_regions,

    const char* input_tensor_name,
    size_t input_layer_size,
    const char* output_tensor_name,
    size_t output_layer_size,

    char** tensor_names,
    size_t* tensor_relative_addrs,
    size_t* tensor_regions,
    size_t* tensor_sizes,
    float* tensor_scales,
    int* tensor_zero_points,
    size_t num_tensors,


    //float in_spk_scale,
    //int in_spk_zero_point,

    //float v_mem_scale,
    //int v_mem_zero_point,
    //float time_not_updated_scale,
    //int time_not_updated_zero_point,

    //float out_spk_scale,
    //int out_spk_zero_point
    int is_last_layer
) {


    //NNLayer* nnlayer = NNLayer_Init(num_tensors, num_regions);
    //if (nnlayer == NULL) { printf("Error when initializing NN_layer0\n"); }


    nnlayer->command_stream = command_stream;
    nnlayer->command_stream_length = command_stream_length;



    // Assign memory regions
    for (size_t i = 0; i < num_regions; i++) {
        nnlayer->memory_regions[memory_region_region_numbers[i]]->region_start_ptr = memory_region_ptrs[i];
        nnlayer->memory_regions[memory_region_region_numbers[i]]->length = memory_region_sizes[i];
    }
    nnlayer->num_regions = num_regions;


    
        




    // Assign Tensors
    nnlayer->num_tensors = num_tensors;
    for (size_t i = 0; i < num_tensors; i++) {
        Tensor_Assign(
            nnlayer->tensors[i],
            tensor_names[i],
            memory_region_ptrs[tensor_regions[i]],
            tensor_regions[i],
            tensor_relative_addrs[i],
            tensor_sizes[i],
            tensor_scales[i],
            tensor_zero_points[i]
        );
        
        /* TEMPORARY SOLUTION !!!! */
        Tensor* tensor = nnlayer->tensors[i];
        // If its V_mem or time not updated, assign 0 to them
        if (strcmp(tensor->name, "V_MEM") == 0 || strcmp(tensor->name, "TIME_NOT_UPDATED") == 0) {
            quantize_float_scalar_to_int8_array(TENSOR_INIT_VALUE, tensor->ptr, tensor->size, tensor->scale, tensor->zero_point);
            //printf("v_mem or time_not_updated\n");
        } else if (strcmp(tensor->name, input_tensor_name) == 0) {  // Assign layer input and output (so other layers know where to read and write from)
            nnlayer->input = tensor;
            //printf("in_spk ptr: %p\n", in_spk);
            Tensor_Print(tensor);
        } else if (strcmp(tensor->name, output_tensor_name) == 0) {
            nnlayer->output = tensor;
            //printf("out_spk: %p\n", out_spk);
            Tensor_Print(tensor);
        } else if (strcmp(tensor->name, "UPDATE_NXT_LAYER") == 0) {
            nnlayer->update_nxt_layer = tensor;
            //printf("update_nxt_layer\n");
        }

        //Tensor_Print(tensor);

    }




    // Assign default value to -1
    nnlayer->time_of_previous_update = -1;


    return 0;


}

Tensor* NNLayer_Get_Tensor(NNLayer* nnlayer, const char* tensor_name) {
    for (size_t i = 0; i < nnlayer->num_tensors; i++) {
        Tensor* tensor = nnlayer->tensors[i];
        if (strcmp(tensor->name, tensor_name) == 0) {
            return tensor;
        }
    }
}




// Assign a tensor to a specific element in the NNLayer
int NNLayer_Assign_Tensor(NNLayer* layer, size_t element, int8_t* region_ptr, size_t region_number, size_t relative_addr, size_t tensor_size, 
                   float scale, int zero_point, const char* tensor_name) {

    // Check if layer is valid
    if (!layer) {
        return -1;  // Invalid layer
    }
    
    // Check if element index is valid
    if (element >= layer->num_tensors) {
        return -2;  // Invalid element index
    }
    
    // Give a warning if previous tensor already exist
    if (layer->tensors[element]->ptr != NULL) {
        printf("Warning: Assigning Tensor to already existing one\n");
        return -3;
    }


    Tensor_Assign(layer->tensors[element], tensor_name, layer->memory_regions[region_number]->region_start_ptr, region_number, relative_addr, tensor_size, scale, zero_point);

    //layer->tensors[element].ptr = tensor_ptr;
    //layer->tensors[element].size = tensor_size;
    //layer->tensors[element].name = tensor_name;
    //layer->tensors[element].scale = scale;
    //layer->tensors[element].zero_point = zero_point;
    //layer->tensors[element].scale_reciprocal = 1 / scale;



    // Free previous tensor if it exists
    //if (layer->tensors[element].ptr != NULL) {
        //free(layer->tensors[element]);
    //}
    
    //// Free previous name if it exists
    //if (layer->tensor_names[element] != NULL) {
        //free(layer->tensor_names[element]);
    //}

    //// Assign new tensor pointer
    //layer->tensor_ptrs[element] = tensor_ptr;
    
    //// Set tensor size
    //layer->tensor_sizes[element] = tensor_size;
    
    //// Set quantization parameters
    //layer->quant_params[element].scale = scale;
    //layer->quant_params[element].zero_point = zero_point;
    //layer->quant_params[element].scale_reciprocal = 1/scale;
    
    //// Allocate and copy tensor name
    //if (tensor_name) {
        //layer->tensor_names[element] = strdup(tensor_name);
        //if (!layer->tensor_names[element]) {
            //return -3;  // Memory allocation failed for name
        //}
    //} else {
        //// Provide default name if none is specified
        //char default_name[32];
        //snprintf(default_name, sizeof(default_name), "tensor_%zu", element);
        //layer->tensor_names[element] = strdup(default_name);
        //if (!layer->tensor_names[element]) {
            //return -3;  // Memory allocation failed for name
        //}
    //}
    
    return 0;  // Success
}

// Free all memory associated with an NNLayer
void NNLayer_Free(NNLayer* layer) {
    if (!layer) {
        return;  // Nothing to free
    }
    

    
    // Free arrays
    free(layer->tensors);
    free(layer->memory_regions);

    
    // Free the layer structure itself
    free(layer);

    // Set assign it to NULL to make sure we dont try to access it later:
    layer = NULL;
}



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
        if (layer->tensors[tensor_idx]->ptr == NULL) {
            printf("  Tensor %zu (%s): NULL\n", 
                  tensor_idx, 
                  layer->tensors[tensor_idx]->name ? layer->tensors[tensor_idx]->name : "unnamed");
            continue;
        }
        
        // Get tensor data and parameters
        int8_t* tensor_data = layer->tensors[tensor_idx]->ptr;
        size_t tensor_size = layer->tensors[tensor_idx]->size;
        float scale = layer->tensors[tensor_idx]->scale;
        int zero_point = layer->tensors[tensor_idx]->zero_point;
        const char* tensor_name = layer->tensors[tensor_idx]->name ? 
                                  layer->tensors[tensor_idx]->name : "unnamed";
        
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
NN_Model* NN_Model_Init(int8_t* total_arena_tensor, NNLayer* first_nnlayer, size_t input_size, size_t output_size, size_t num_time_steps, size_t out_spk_sum_tensor_idx) {
    // Init Total Tensor Arena and store its pointer
    NN_Model* nn_model = (NN_Model*)malloc(sizeof(NN_Model));


    // Set number of time steps
    nn_model->num_time_steps = num_time_steps;
    
    
    //note: remove total_tensor_arena as attribute from NN_Model struct
    //nn_model->total_tensor_arena = total_arena_tensor;
    nn_model->first_nnlayer = first_nnlayer;

    // Set nn model input tensor
    nn_model->input = first_nnlayer->input;

    // Set nn model output tensor
    NNLayer* nnlayer = first_nnlayer;
    size_t num_layers = 0;
    while (nnlayer->next_layer != NULL) {
        nnlayer = nnlayer->next_layer;
        num_layers += 1;
    }

    printf("num layers: %d\n", num_layers);
    nn_model->output = nnlayer->output;
    nn_model->num_layers = num_layers;

    // Set last_layer
    nn_model->last_nnlayer = nnlayer;

    // Set out_spk_sum pointer to the out_spk_sum of the last layer 
    //nn_model->out_spk_sum = nnlayer->tensor_ptrs[OUT_SPK_SUM_TENSOR_IDX];
    nn_model->out_spk_sum = NNLayer_Get_Tensor(nn_model->last_nnlayer, "OUT_SPK_SUM");






    return nn_model;


}
