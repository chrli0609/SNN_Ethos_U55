#pragma once

#include <stdint.h> //int8_t
#include <stddef.h> //size_t



#define MEM_ALIGNMENT 16


// Simple Persistent Buffer Allocator (like PersistentArenaBufferAllocator)
typedef struct {
    int8_t* buffer_head;  // Start of buffer
    int8_t* tail_temp;    // Current tail position
} PersistentAllocator;


// Updated NNLayer structure to include tensor names
typedef struct {
    float scale;
    int zero_point;
} tuple;

typedef struct {
    int8_t* tensor_arena;
    size_t tensor_arena_size;

    int8_t** tensor_ptrs;   // Array of int8_t pointers
    size_t* tensor_sizes;   // Array of tensor sizes
    tuple* quant_params;    // Array of quantization parameter tuples
    char** tensor_names;    // Array of tensor names
    size_t num_tensors;     // Number of tensors

    PersistentAllocator allocator;


    int8_t* input;
    int8_t* update_nxt;
    int8_t* output;
    struct NNLayer* next_layer;
} NNLayer;

typedef struct {
    int8_t* total_tensor_arena;
    NNLayer* first_nnlayer;
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