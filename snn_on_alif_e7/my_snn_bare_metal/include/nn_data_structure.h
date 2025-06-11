#pragma once

#include <stdint.h> //int8_t
#include <stddef.h> //size_t



#define MEM_ALIGNMENT 16


// Simple Persistent Buffer Allocator (like PersistentArenaBufferAllocator)
typedef struct {
    int8_t* buffer_head;  // Start of buffer
    int8_t* tail_temp;    // Current tail position
} PersistentAllocator;


// Const tensors for each FC_LIF_Layer
typedef struct {
    const uint8_t* command_stream;
    size_t command_stream_length;

    const int8_t* bias_and_weights;
    size_t bias_and_weights_length;

    const int8_t* lif_params;
    size_t lif_params_length;

    const int8_t* luts;
    size_t luts_length;


} FC_LIF_Const_Param;

// Updated NNLayer structure to include tensor names
typedef struct {
    float scale;
    int zero_point;
    float scale_reciprocal;
} tuple;

typedef struct {

    // The tensor where most of the memory allocated for the NPU is stored
    // (i.e. bias, weights, tmp1, tmp2, v_mem, time_not_updated, update_nxt_layer)
    int8_t* tensor_arena;
    size_t tensor_arena_size;

    // Keeps track of the non-const memory for each layer
    int8_t** tensor_ptrs;   // Array of int8_t pointers
    size_t* tensor_sizes;   // Array of tensor sizes
    tuple* quant_params;    // Array of quantization parameter tuples
    char** tensor_names;    // Array of tensor names
    size_t num_tensors;     // Number of tensors


    PersistentAllocator allocator;

    // Pointers to the const tensors (i.e. weights+bias, lif_params (ln_beta & vth), lut)
    FC_LIF_Const_Param fc_lif_const_tensors;

    float time_of_previous_update;


    int8_t* input;
    int8_t* update_nxt;
    int8_t* update_curr;
    int8_t* output;
    struct NNLayer* next_layer;
} NNLayer;

typedef struct {
    size_t input_size;
    size_t output_size;


    int8_t* total_tensor_arena;
    NNLayer* first_nnlayer;
    NNLayer* last_nnlayer;
    size_t num_layers;

    int8_t* input;
    int8_t* output;
    int8_t* out_spk_sum;
} NN_Model;


void* aligned_malloc(size_t required_bytes, size_t alignment);
void aligned_free(void* p);

void PersistentAllocator_Init(PersistentAllocator* allocator, int8_t* arena, size_t size);
void* PersistentAllocator_GetAbsPointer(PersistentAllocator* allocator, size_t relative_addr);
void* PersistentAllocator_Allocate(PersistentAllocator* allocator, size_t size, size_t alignment);


int8_t* PersistentAllocator_GetBufferHead(PersistentAllocator* allocator);
int8_t* PersistentAllocator_GetTailTemp(PersistentAllocator* allocator);

NNLayer* NNLayer_Init(int8_t* tensor_arena, size_t tensor_arena_size, size_t num_tensors);
int NNLayer_Assign(NNLayer* layer, size_t element, int8_t* tensor_ptr, size_t tensor_size, 
                   float scale, int zero_point, const char* tensor_name);
void NNLayer_Free(NNLayer* layer);
void NNLayer_DequantizeAndPrint(const NNLayer* layer);


NN_Model* NN_Model_Init(int8_t* total_arena_tensor, NNLayer* first_nnlayer);