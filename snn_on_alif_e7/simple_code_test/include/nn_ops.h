#pragma once



#include "lif_model.h"
#include <stdint.h>
#include <stddef.h>






// Simple Persistent Buffer Allocator (like PersistentArenaBufferAllocator)
typedef struct {
    int8_t* buffer_head;  // Start of buffer
    int8_t* tail_temp;    // Current tail position
} PersistentAllocator;



// Initializes the PersistentAllocator
void PersistentAllocator_Init(PersistentAllocator* allocator, int8_t* arena, size_t size);

// Allocates a persistent buffer
void* PersistentAllocator_Allocate(PersistentAllocator* allocator, size_t size, size_t alignment);

// Manually allocate relative addressing
void* PersistentAllocator_GetAbsPointer(PersistentAllocator* allocator, size_t relative_addr);

















int my_mem_u_npu(
    int8_t* tensor_arena,
    size_t tensor_arena_size,

    //int8_t* in_spk,
    //size_t in_spk_tensor_size,
    //int8_t* v_mem,
    //size_t v_mem_tensor_size,
    //int8_t* decay,
    //size_t decay_tensor_size,

    //relative addressing
    //size_t in_spk_rel_addr,
    //size_t out_spk_rel_addr,

    

    const uint8_t* command_stream,
    size_t command_stream_size,
    const int8_t* weight_tensor,
    size_t weight_tensor_size,


    const int8_t* exp_lut,
    size_t exp_lut_size


    //int8_t* v_mem_new,
    //size_t v_mem_new_tensor_size,
    //int8_t* out_spk,
    //size_t out_spk_tensor_size
);



int membrane_update_npu(
    size_t tensor_arena_size,
    int8_t* in_spk,
    size_t in_spk_tensor_size,
    int8_t* v_mem,
    size_t v_mem_tensor_size,
    int8_t* decay,
    size_t decay_tensor_size,
    
    

    const uint8_t* command_stream,
    size_t command_stream_size,
    const int8_t* weight_tensor,
    size_t weight_tensor_size,



    int8_t* v_mem_new,
    size_t v_mem_new_tensor_size,
    int8_t* out_spk,
    size_t out_spk_tensor_size
);

int elementwise_mul(
    size_t tensor_arena_size,
    int8_t* input1,
    size_t input1_tensor_size,
    int8_t* input2,
    size_t input2_tensor_size,

    const uint8_t* command_stream,
    size_t command_stream_size,
    const uint8_t* scales_tensor,
    size_t scales_tensor_size,

    int8_t* output,
    size_t output_tensor_size
);



int matmul(uint8_t* input, uint8_t* output);


