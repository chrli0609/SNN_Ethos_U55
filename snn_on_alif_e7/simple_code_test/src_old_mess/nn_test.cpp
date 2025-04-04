#include <stdint.h>
#include <stddef.h>


#include <stdio.h>
#include <string.h>


#include <cstdlib> //malloc

#include "ethosu_driver.h"


//try using tensorflow memory allocator
//include "tensorflow/lite/micro/micro_allocator.h"
#include "micro_allocator.h"



//#include "conv2d_model.hpp"
/*
int create_tensors(int num_tensors, size_t* base_addrs_size, uint64_t* base_addrs) {
    size_t alignment = 16;  // Must be a power of 2

    for (size_t i = 0; i < num_tensors; i++) {
        uint8_t* ptr = (uint8_t*)aligned_alloc(alignment, base_addrs_size[i] * sizeof(uint8_t));
        
        if (ptr == NULL) {
            printf("Memory allocation failed for tensor %zu!\n", i);
            return -1;
        }

        base_addrs[i] = (uint64_t)(uintptr_t)ptr;
    }

    return 0;
}

// Modified function to assign values to only one specific tensor at a time
void assign_tensor_values(int tensor_index, size_t* base_addrs_size, uint64_t* base_addrs, uint8_t val_to_insert) {
    uint8_t* tensor_data = (uint8_t*)(uintptr_t)base_addrs[tensor_index];

    printf("about to assign value\n");
    for (size_t j = 0; j < base_addrs_size[tensor_index]; j++) {
        tensor_data[j] = val_to_insert; // Use the provided value
    }
    printf("finished assigning value\n");
}

void print_tensors(int tensor_index, size_t* base_addrs_size, uint64_t* base_addrs) {
    uint8_t* tensor_data = (uint8_t*)(uintptr_t)base_addrs[tensor_index];

    printf("Tensor %d first 5 values: ", tensor_index);
    for (int j = 0; j < 5 && j < base_addrs_size[tensor_index]; j++) {
        printf("%u ", tensor_data[j]);
    }
    printf("\n");
}

void free_tensors(int num_tensors, uint64_t* base_addrs) {
    for (int i = 0; i < num_tensors; i++) {
        free((void*)(uintptr_t)base_addrs[i]); 
    }
}
*/


// Aligns a pointer down to the nearest aligned address
void* AlignPointerDown(void* ptr, size_t alignment) {
    return reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(ptr) & ~(alignment - 1));
}

// Simple Persistent Buffer Allocator (like PersistentArenaBufferAllocator)
class PersistentAllocator {
 public:
  PersistentAllocator(uint8_t* arena, size_t size)
      : buffer_head_(arena), tail_temp_(arena + size) {}

  void* AllocatePersistentBuffer(size_t size, size_t alignment) {
    uint8_t* aligned_result = static_cast<uint8_t*>(
        AlignPointerDown(tail_temp_ - size, alignment));
    
    if (aligned_result < buffer_head_) {
      printf("Memory allocation failed! Requested: %zu bytes\n", size);
      return nullptr;
    }

    tail_temp_ = aligned_result;
    return aligned_result;
  }

 private:
  uint8_t* buffer_head_;  // Start of buffer
  uint8_t* tail_temp_;    // Current tail position
};

// Function to print uint8_t tensor values
void PrintTensor(uint8_t* tensor, size_t num_elements) {
    if (!tensor) {
      printf("Tensor is NULL!\n");
      return;
    }
  
    printf("Tensor values: ");
    for (size_t i = 0; i < num_elements; i++) {
      printf("%u ", tensor[i]);
    }
    printf("\n");
  }



int create_n_run_cmd_stream() {

    //Get length of command stream
    const uint8_t* command_stream = GetModelPointer();
    size_t command_stream_size = GetModelLen();
    



    printf("whefoawhoiawoehaw\n");

    // This code will assign the NPU base addresses like this:
    //
    // +--------------+----------------------+
    // | Base address | Description          |
    // +--------------+----------------------+
    // |            0 | TFLM model           |
    // |            1 | TFLM arena           |
    // |            2 | Ethos-U fast scratch |
    // |         3..n | Input tensors        |
    // |         n..m | Output tensors       |
    // +--------------+----------------------+
    //
    // The number of base address will be limited to 8.
    //
    // NOTE! The command stream produced by Vela will access the IFM and OFM
    // buffers using base address 1. This means that it is not possible to point
    // the input and output tensors outside of the TFLM arena.
    
    // In preloaded Yolo_v4 example
    //Base_addr[0]: main_split_1_flash
    //Base_addr[1]: main_split_1_scratch
    //Base_addr[2]: main_split_1_scratch_fast
    //Base_addr[3]: image_input     (input)
    //Base_addr[4]: identity_1      (output 1)
    //Base_addr[5]: identity        (output 2)


    //int num_tensors = 5;
    //uint64_t base_addrs[num_tensors]; 
    //size_t base_addrs_size[num_tensors] = {
    //    0,
    //    2048,
    //    2048,
    //    8 * 8 * 16,  
    //    4 * 4 * 16   
    //};
//
//
    //printf("before create_tensors\n");
    //if (create_tensors(num_tensors, base_addrs_size, base_addrs) != 0) {
    //    return 1;
    //}
    //printf("after create_tensors\n");
//
    //// Assign values to only one tensor at a time
    //assign_tensor_values(0, base_addrs_size, base_addrs, 0); // Fill second tensor with 99
    ////printf("tensor 0 values assigned\n");
    //assign_tensor_values(1, base_addrs_size, base_addrs, 0); // Fill second tensor with 99
    ////printf("tensor 1 values assigned\n");
    //assign_tensor_values(2, base_addrs_size, base_addrs, 0); // Fill second tensor with 99
    ////printf("tensor 2 values assigned\n");
    //assign_tensor_values(3, base_addrs_size, base_addrs, 0); // Fill first tensor with 42
    ////printf("tensor 3 values assigned\n");
    //assign_tensor_values(4, base_addrs_size, base_addrs, 0); // Fill second tensor with 99
//
    //printf("tensor values assigned\n");
//
    //// Print to verify
    //printf("Before Inference!\n");
    //for (size_t i = 0; i < num_tensors; i++) {
    //    print_tensors(i, base_addrs_size, base_addrs);
    //}
    




    // seems like we need to use model for AllocatePersistrentTfLiteTensor
    // try with tflite interpreter and do memory planning maybe its possible to
    // reuse this part, hopefully
    //with tflite interpreter

    const int tensor_arena_size = 2048;
    uint8_t tensor_arena[tensor_arena_size] __attribute__((aligned(16)));
    size_t ALIGNMENT = 16;

    size_t INPUT_SIZE = 1*8*8*16;
    size_t OUTPUT_SIZE = 1*4*4*16;

    PersistentAllocator allocator(tensor_arena, tensor_arena_size);


    // Allocate for input tensor
    uint8_t* input_tensor = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(INPUT_SIZE*sizeof(uint8_t), ALIGNMENT));
    if (input_tensor) {
        for (int i = 0; i < INPUT_SIZE; i++) {
          input_tensor[i] = 13;  // Writing float values
        }
    }

    // Allocate output tensor
    //uint8_t* output_tensor = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(OUTPUT_SIZE*sizeof(uint8_t), ALIGNMENT));
    //if (output_tensor) {
    //    for (int i = 0; i < OUTPUT_SIZE; i++) {
    //      output_tensor[i] = 0;  // Writing float values
    //    }
    //}
    

    //uint8_t* output_tensor = (uint8_t*)reinterpret_cast<uintptr_t>(tensor_arena);


    // print values
    printf("BEFORE INVOKE\n");
    printf("tensor_arena\n");
    PrintTensor(tensor_arena, tensor_arena_size);
    printf("input_tensor\n");
    PrintTensor(input_tensor, INPUT_SIZE);
    //printf("output_tensor\n");
    //PrintTensor(output_tensor, OUTPUT_SIZE);
    printf("actual output_tensor\n");
    PrintTensor(tensor_arena, OUTPUT_SIZE);











    // No model weights for MAXPOOL
    uint8_t model_weight_tensor[0];


    int num_tensors = 4;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(model_weight_tensor));    //model weights
    base_addrs[1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));           //Tensor arena pointer
    base_addrs[2] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));           //Fast scratch, just keep same as tensor arena for now
    base_addrs[3] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_tensor));           // Input tensor (lies in the tensor arena)
    //base_addrs[4] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(output_tensor));          // Output tensor (lies in the tensor arena)
    base_addrs[4] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));

    base_addrs_size[0] = 0;
    base_addrs_size[1] = tensor_arena_size;
    base_addrs_size[2] = tensor_arena_size;
    base_addrs_size[3] = INPUT_SIZE;
    base_addrs_size[4] = OUTPUT_SIZE;


    

    
    


    //mydebug
    //for (int i = 0; i < num_tensors; i++) {
    //    printf("base_addrs[%d] = %p\n", i, base_addrs[i]);
    //}


    


    // Reserve the Ethos-U driver
    struct ethosu_driver* drv = ethosu_reserve_driver();
    if (!drv) {
        printf("Failed to reserve Ethos-U driver\n");
        return -1;
    }

    printf("Before invoke_v3: &drv = %p, drv = %p\n", (void*)&drv, (void*)drv);


    //test_npu_driver(drv, command_stream, command_stream_size, 
    //    base_addrs, base_addrs_size, num_tensors, NULL);

    int result = ethosu_invoke_v3(drv, command_stream, command_stream_size, 
        base_addrs, base_addrs_size, num_tensors, NULL);


    //struct ethosu_driver*const new_drv = (struct ethosu_driver*)536879144;

    printf("After invoke_v3: &drv = %p, drv = %p\n", (void*)&drv, (void*)drv);



    printf("result: %d\n", result);
    if (result == -1) {
        printf("ERRORRRRRR\n");
        return -1;
    } else {
        printf("its all okk\n");
    }
    


    ethosu_release_driver(drv);

    printf("Driver release successfully\n");



    // print values
    printf("AFTER INVOKE\n");
    printf("tensor_arena\n");
    PrintTensor(tensor_arena, tensor_arena_size);
    printf("input_tensor\n");
    PrintTensor(input_tensor, INPUT_SIZE);
    //printf("output_tensor\n");
    //PrintTensor(output_tensor, OUTPUT_SIZE);
    printf("actual output_tensor\n");
    PrintTensor(tensor_arena, OUTPUT_SIZE);

    // Print results
    //size_t input_size = sizeof(base_addrs[0]) / sizeof((base_addrs[0])[0]);
    //size_t weight_size = sizeof(base_addrs[1]) / sizeof((base_addrs[1])[0]);
    //size_t output_size = sizeof(base_addrs[2]) / sizeof((base_addrs[2])[0]);


    //printf("input_data size: %d\n", input_size);
    //printf("input_data size: %d\n", base_addrs_size[0]);
    //print_uint8_array(base_addrs[0], base_addrs_size[0]);

    //printf("weight_data_size: %d\n", base_addrs_size[1]);
    //print_uint8_array(base_addrs[1], base_addrs_size[1]);

    //printf("output_data_size: %d\n", base_addrs_size[2]);
    //print_uint8_array(base_addrs[2], base_addrs_size[2]);
    


    //printf("After Inference!\n");
    //for (size_t i = 0; i < num_tensors; i++) {
    //    print_tensors(i, base_addrs_size, base_addrs);
    //}

    ////free the tensors
    //free_tensors(num_tensors, base_addrs);


    //printf("Tensors freed successfully\n");



    printf("Still in create_n_run_cmd_stream\n");

    return 0;

}
