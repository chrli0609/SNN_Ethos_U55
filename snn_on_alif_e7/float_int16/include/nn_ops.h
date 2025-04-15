#pragma once



#include "lif_model.h"
#include <stdint.h>
#include <stddef.h>






// Simple Persistent Buffer Allocator (like PersistentArenaBufferAllocator)
typedef struct {
    int16_t* buffer_head;  // Start of buffer
    int16_t* tail_temp;    // Current tail position
} PersistentAllocator;



// Initializes the PersistentAllocator
void PersistentAllocator_Init(PersistentAllocator* allocator, int16_t* arena, size_t size);

// Allocates a persistent buffer
void* PersistentAllocator_Allocate(PersistentAllocator* allocator, size_t size, size_t alignment);

// Manually allocate relative addressing
void* PersistentAllocator_GetAbsPointer(PersistentAllocator* allocator, size_t relative_addr);












int my_mem_u_npu(
    int16_t* tensor_arena,
    size_t tensor_arena_size,
    

    const uint8_t* command_stream,
    size_t command_stream_size,
    const int8_t* weight_tensor,
    size_t weight_tensor_size

);



