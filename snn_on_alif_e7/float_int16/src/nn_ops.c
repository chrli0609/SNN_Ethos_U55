#include "include/nn_ops.h"






#include <stdio.h>
#include <stddef.h>




#include "ethosu_driver.h"


#include "include/extra_funcs.h"




#define MEM_ALIGNMENT 16


extern int DEBUG_MODE;







// Aligns a pointer down to the nearest aligned address
void* AlignPointerDown(void* ptr, size_t alignment) {
    return (void*)((uintptr_t)ptr & ~(alignment - 1));
}

// Initializes the PersistentAllocator
void PersistentAllocator_Init(PersistentAllocator* allocator, int16_t* arena, size_t size) {
    allocator->buffer_head = arena;
    allocator->tail_temp = arena + size;
}

// Allocates a persistent buffer
void* PersistentAllocator_Allocate(PersistentAllocator* allocator, size_t size, size_t alignment) {
    int16_t* aligned_result = (int16_t*)AlignPointerDown(allocator->tail_temp - size, alignment);
    
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
int16_t* PersistentAllocator_GetBufferHead(PersistentAllocator* allocator) {
    return allocator->buffer_head;
}

int16_t* PersistentAllocator_GetTailTemp(PersistentAllocator* allocator) {
    return allocator->tail_temp;
}







int run_cms(
    const uint8_t* command_stream,
    size_t command_stream_size,
    uint64_t* base_addrs,
    size_t* base_addrs_size,
    int num_tensors

) {


    // Reserve the Ethos-U driver
    struct ethosu_driver* drv = ethosu_reserve_driver();
    if (!drv) {
        printf("Failed to reserve Ethos-U driver\n");
        return -1;
    }

    if (DEBUG_MODE) { printf("Before invoke_v3: &drv = %p, drv = %p\n", (void*)&drv, (void*)drv); }

    if(ethosu_invoke_v3(drv, command_stream, command_stream_size, 
        base_addrs, base_addrs_size, num_tensors, NULL) != 0) {
        printf("Invoke_v3 Failed\n");
        return -1;
    }
    //struct ethosu_driver*const new_drv = (struct ethosu_driver*)536879144;

    //mydebug
    if (DEBUG_MODE) { printf("After invoke_v3: &drv = %p, drv = %p\n", (void*)&drv, (void*)drv); }


    ethosu_release_driver(drv);

    if (DEBUG_MODE) { printf("Driver release successfully\n"); }



    return 0;

}






int my_mem_u_npu(
    int16_t* tensor_arena,
    size_t tensor_arena_size,


    const uint8_t* command_stream,
    size_t command_stream_size,
    const int8_t* weight_tensor,
    size_t weight_tensor_size
)
{






    if (DEBUG_MODE) {

        // print values
        printf("BEFORE INVOKE\n");
        PrintInt16Tensor("tensor_arena", tensor_arena, tensor_arena_size);
    }



    // Assign base addrs
    const size_t num_tensors = 3;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] = (uint64_t)(intptr_t)weight_tensor;   // Model weights
    base_addrs[1] = (uint64_t)(intptr_t)tensor_arena;   // Tensor arena pointer
    base_addrs[2] = (uint64_t)(intptr_t)tensor_arena;   // Fast scratch, same as tensor arena for now
    //base_addrs[3] = (uint64_t)(intptr_t)input_tensor;   // Input tensor (in tensor arena)
    //base_addrs[4] = (uint64_t)(intptr_t)output_tensor;  

    base_addrs_size[0] = weight_tensor_size;
    base_addrs_size[1] = tensor_arena_size;
    base_addrs_size[2] = tensor_arena_size;
    //base_addrs_size[3] = input_tensor_size;
    //base_addrs_size[4] = output_tensor_size;




    // Run NPU commands
    if(run_cms(command_stream, command_stream_size, base_addrs, base_addrs_size, num_tensors) != 0) {
        printf("run_cms call failed\n");
        return -1;
    }


    if (DEBUG_MODE) {
        //print tensor values after
        printf("AFTER INVOKE\n");
        PrintInt16Tensor("tensor_arena", tensor_arena, tensor_arena_size);
        
    }


    return 0;




}