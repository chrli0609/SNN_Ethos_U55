#include <stdint.h>
#include <stddef.h>


#include <stdio.h>
#include <string.h>


#include <cstdlib> //malloc

#include "ethosu_driver.h"


//try using tensorflow memory allocator
//include "tensorflow/lite/micro/micro_allocator.h"
#include "micro_allocator.h"



//#define MEM_ALIGNMENT 16

extern const size_t MEM_ALIGNMENT = 16;


//#include "conv2d_model.hpp"



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

    printf("Before invoke_v3: &drv = %p, drv = %p\n", (void*)&drv, (void*)drv);


    if(ethosu_invoke_v3(drv, command_stream, command_stream_size, 
        base_addrs, base_addrs_size, num_tensors, NULL) != 0) {
        printf("Invoke_v3 Failed\n");
        return -1;
    } else {
        printf("Invoke_v3 called successfully\n");
    }
    
    //struct ethosu_driver*const new_drv = (struct ethosu_driver*)536879144;

    printf("After invoke_v3: &drv = %p, drv = %p\n", (void*)&drv, (void*)drv);


    ethosu_release_driver(drv);
    printf("Driver release successfully\n");



    return 0;

}




int maxpool2d(size_t input_size, size_t output_size) {
    //Get length of command stream
    const uint8_t* command_stream = GetMaxPool2DPointer();
    size_t command_stream_size = GetMaxPool2DLen();

    const int tensor_arena_size = 2048;
    uint8_t tensor_arena[tensor_arena_size] __attribute__((aligned(16)));

    // Allocate Tensor Arena
    PersistentAllocator allocator(tensor_arena, tensor_arena_size);


    // Allocate for input tensor
    uint8_t* input_tensor = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(input_size*sizeof(uint8_t), MEM_ALIGNMENT));
    if (input_tensor) {
        for (int i = 0; i < input_size; i++) {
          input_tensor[i] = 13;  // Writing float values
        }
    }


    // print values
    printf("BEFORE INVOKE\n");
    printf("tensor_arena\n");
    PrintTensor(tensor_arena, tensor_arena_size);
    printf("input_tensor\n");
    PrintTensor(input_tensor, input_size);
    //printf("output_tensor\n");
    //PrintTensor(output_tensor, OUTPUT_SIZE);
    printf("actual output_tensor\n");
    PrintTensor(tensor_arena, output_size);



    // No model weights for MAXPOOL
    uint8_t model_weight_tensor[0] __attribute__((aligned(16)));


    
    int num_tensors = 5;
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
    base_addrs_size[3] = input_size;
    base_addrs_size[4] = output_size;


    if(run_cms(command_stream, command_stream_size, base_addrs, base_addrs_size, num_tensors) != 0) {
        printf("run_cms call failed\n");
        return -1;
    } else {
        printf("run_cms called successfully\n");
    }




        // print values
        printf("AFTER INVOKE\n");
        printf("tensor_arena\n");
        PrintTensor(tensor_arena, tensor_arena_size);
        printf("input_tensor\n");
        PrintTensor(input_tensor, input_size);
        //printf("output_tensor\n");
        //PrintTensor(output_tensor, OUTPUT_SIZE);
        printf("actual output_tensor\n");
        PrintTensor(tensor_arena, output_size);


    return 0;
}


int conv2d(size_t input_size, size_t output_size)
{

    //Get length of command stream
    const uint8_t* command_stream = GetConv2DPointer();
    size_t command_stream_size = GetConv2DLen();

    //const int tensor_arena_size = 3248;
    const int tensor_arena_size = 3248;
    uint8_t tensor_arena[tensor_arena_size] __attribute__((aligned(16)));

    // Allocate Tensor Arena
    PersistentAllocator allocator(tensor_arena, tensor_arena_size);


    // Allocate for input tensor
    uint8_t* input_tensor = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(input_size*sizeof(uint8_t), MEM_ALIGNMENT));
    if (input_tensor) {
        for (int i = 0; i < input_size; i++) {
          input_tensor[i] = 0;  // Writing float values
        }
    }



     // Set weights for Conv2d to 0
    //uint8_t model_weight_tensor[weight_size] __attribute__((aligned(16))) = { 0 };
    //for (size_t i = 0; i < weight_size; i++) {
    //    model_weight_tensor[i] = 0;
    //}

    // Allocate for weight tensor
    //uint8_t* weight_tensor = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(weight_size*sizeof(uint8_t), MEM_ALIGNMENT));
    //if (weight_tensor) {
    //    for (int i = 0; i < weight_size; i++) {
    //        weight_tensor[i] = 0;  // Writing float values
    //    }
    //}


    //Get weight tensor pointer and length
    const uint8_t* weight_tensor = GetConv2DWeightsPointer();
    size_t weight_size = GetConv2DWeightsLen();

    // print values
    printf("BEFORE INVOKE\n");
    printf("tensor_arena\n");
    PrintTensor(tensor_arena, tensor_arena_size);
    printf("input_tensor\n");
    PrintTensor(input_tensor, input_size);
    //printf("output_tensor\n");
    //PrintTensor(output_tensor, OUTPUT_SIZE);
    printf("actual output_tensor\n");
    PrintTensor(tensor_arena, output_size);



    printf("weight_tensor: %p\n", weight_tensor);
    //PrintTensor(model_weight_tensor, weight_size);
    printf("Tensor values: ");
    for (size_t i = 0; i < weight_size; i++) {
      printf("0x%02x ", weight_tensor[i]);
    }
    printf("\n");

    

    
    int num_tensors = 5;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(weight_tensor));    //model weights
    base_addrs[1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));           //Tensor arena pointer
    base_addrs[2] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));           //Fast scratch, just keep same as tensor arena for now
    base_addrs[3] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_tensor));           // Input tensor (lies in the tensor arena)
    //base_addrs[4] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(output_tensor));          // Output tensor (lies in the tensor arena)
    base_addrs[4] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));

    base_addrs_size[0] = weight_size;
    base_addrs_size[1] = tensor_arena_size;
    base_addrs_size[2] = tensor_arena_size;
    base_addrs_size[3] = input_size;
    base_addrs_size[4] = output_size;



    const uint8_t* ptr_after = (uint8_t*)reinterpret_cast<void*>(base_addrs[0]);  // Pointer to the array
    printf("before invoke: weight tensor: %p\n", weight_tensor);
      for (int i = 0; i < base_addrs_size[0]; ++i) {
          printf("0x%02x ", ptr_after[i]);  // Print each byte in hexadecimal format
    }
    printf("\n");
  



    //printf("Values in model_weight_tensor: ");
    //for (size_t i = 0; i < 2; i++) {
    //    printf("%u ", model_weight_tensor[i]); // Print each value as an integer
    //}
    //printf("\n");
//
//
//
    //// Cast back to uint8_t* and print values using printf
    //uint8_t* tensor_ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(base_addrs[0]));
    //
//
    //printf("Values in model_weight_tensor: ");
    //for (size_t i = 0; i < base_addrs_size[0]; i++) {
    //    printf("%u ", tensor_ptr[i]); // Print each value as an integer
    //}
    //printf("\n");




    if(run_cms(command_stream, command_stream_size, base_addrs, base_addrs_size, num_tensors) != 0) {
        printf("run_cms call failed\n");
        return -1;
    } else {
        printf("run_cms called successfully\n");
    }




    // print values
    printf("AFTER INVOKE\n");
    printf("tensor_arena\n");
    PrintTensor(tensor_arena, tensor_arena_size);
    printf("input_tensor\n");
    PrintTensor(input_tensor, input_size);
    //printf("output_tensor\n");
    //PrintTensor(output_tensor, OUTPUT_SIZE);
    printf("actual output_tensor\n");
    PrintTensor(tensor_arena, output_size);


    // print weights
    const uint8_t* ptr_after_invoke = (uint8_t*)reinterpret_cast<void*>(base_addrs[0]);  // Pointer to the array
    printf("after invoke: weight tensor: %p\n", weight_tensor);
      for (int i = 0; i < base_addrs_size[0]; ++i) {
          printf("0x%02x ", ptr_after_invoke[i]);  // Print each byte in hexadecimal format
    }
    printf("\n");


    return 0;
    


}