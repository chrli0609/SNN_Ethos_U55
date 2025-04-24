#include "include/nn_ops.h"






#include <stdio.h>
#include <stddef.h>




#include "ethosu_driver.h"
//#include "include/extra_funcs.h"






extern int DEBUG_MODE;














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








/*
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
    PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
    PrintTensor("input_tensor", input_tensor, input_size);
    //printf("output_tensor\n");
    //PrintTensor(output_tensor, OUTPUT_SIZE);
    PrintTensor("output_tensor", tensor_arena, output_size);



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
        PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
        PrintTensor("input_tensor", input_tensor, input_size);
        //printf("output_tensor\n");
        //PrintTensor(output_tensor, OUTPUT_SIZE);
        PrintTensor("output_tensor", tensor_arena, output_size);


    return 0;
}


int conv2d(size_t input_size, size_t output_size)
{

    //Get length of command stream
    const uint8_t* command_stream = Getconv2dCMSPointer();
    size_t command_stream_size = Getconv2dCMSLen();
    printf("command_stream_size = %d\n", command_stream_size);

    //const int tensor_arena_size = 3248;
    //const int tensor_arena_size = 1448;
    //const int tensor_arena_size = 1456;
    //const int tensor_arena_size = 2048;
    //const int tensor_arena_size = 1584;
    uint8_t tensor_arena[CONV2D_TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

    // Allocate Tensor Arena
    PersistentAllocator allocator(tensor_arena, CONV2D_TENSOR_ARENA_SIZE);


    // Allocate for input tensor
    uint8_t* input_tensor = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(input_size*sizeof(uint8_t), MEM_ALIGNMENT));
    if (input_tensor) {
        for (int i = 0; i < input_size; i++) {
          input_tensor[i] = 1;  // Writing float values
        }
    }




    // Allocate for output tensor
    uint8_t* output_tensor = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(output_size*sizeof(uint8_t), MEM_ALIGNMENT));
    if (output_tensor) {
        for (int i = 0; i < output_size; i++) {
            output_tensor[i] = 0;  // init to 0
        }
    }



    //Get weight tensor pointer and length
    const uint8_t* weight_tensor = Getconv2dWeightsPointer();
    size_t weight_size = Getconv2dWeightsLen();
    printf("weight_size: %d", weight_size);
    

    // Allocate for weights & biases
    uint8_t* weight_tensor_on_sram = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(weight_size*sizeof(uint8_t), MEM_ALIGNMENT));
    if (weight_tensor_on_sram) {
        for (int i = 0; i < weight_size; i++) {
            weight_tensor_on_sram[i] = 0;  // init to 0
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


    

    // print values
    printf("BEFORE INVOKE\n");
    PrintTensor("tensor_arena", tensor_arena, CONV2D_TENSOR_ARENA_SIZE);
    PrintTensor("input_tensor", input_tensor, input_size);
    PrintTensor("output_tensor", output_tensor, output_size);



    printf("weight_tensor: %p\n", weight_tensor);
    //PrintTensor(model_weight_tensor, weight_size);
    printf("Tensor values: ");
    for (size_t i = 0; i < weight_size; i++) {
      printf("0x%02x ", weight_tensor[i]);
    }
    printf("\n");

    

    
    const size_t num_tensors = 5;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(weight_tensor));    //model weights
    base_addrs[1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));           //Tensor arena pointer
    base_addrs[2] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));           //Fast scratch, just keep same as tensor arena for now
    base_addrs[3] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_tensor));           // Input tensor (lies in the tensor arena)
    base_addrs[4] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(output_tensor));          // Output tensor (lies in the tensor arena)
    //base_addrs[4] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));

    base_addrs_size[0] = weight_size;
    base_addrs_size[1] = CONV2D_TENSOR_ARENA_SIZE;
    base_addrs_size[2] = CONV2D_TENSOR_ARENA_SIZE;
    base_addrs_size[3] = input_size;
    base_addrs_size[4] = output_size;



    //const uint8_t* ptr_after = (uint8_t*)reinterpret_cast<void*>(base_addrs[0]);  // Pointer to the array
    //printf("before invoke: weight tensor: %p\n", weight_tensor);
    //  for (int i = 0; i < base_addrs_size[0]; ++i) {
    //      printf("0x%02x ", ptr_after[i]);  // Print each byte in hexadecimal format
    //}
    //printf("\n");
  



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
    PrintTensor("tensor_arena", tensor_arena, CONV2D_TENSOR_ARENA_SIZE);
    PrintTensor("input_tensor", input_tensor, input_size);
    PrintTensor("output_tensor", output_tensor, output_size);
    //printf("actual output_tensor\n");
    //PrintTensor(tensor_arena, output_size);

    //printf("checking if negative?\n");
    //for (int i = 0; i < output_size; i++) {
    //    printf("%" PRIi8 " : ", (int8_t)tensor_arena[i]);
    //}


    // print weights
    const uint8_t* ptr_after_invoke = (uint8_t*)reinterpret_cast<void*>(base_addrs[0]);  // Pointer to the array
    printf("after invoke: weight tensor: %p\n", weight_tensor);
      for (int i = 0; i < base_addrs_size[0]; ++i) {
          printf("0x%02x ", ptr_after_invoke[i]);  // Print each byte in hexadecimal format
    }
    printf("\n");


    return 0;
    
}



*/

/*
#include "nn_ops/elementwise_add.h"
int elementwise_add()
{

    printf("Just after entering elementwise_add()\n");

    //Check that the submitted tensors have the correct sizes

    printf("Just before calling cms\n");
    //Get length of command stream
    const uint8_t* command_stream = GetElementwiseAddCMSPointer();
    size_t command_stream_size = GetElementwiseAddCMSLen();

    printf("Just after calling cms\n");

    const size_t tensor_arena_size = ELEMENTWISE_ADD_TENSOR_ARENA_SIZE;
    int8_t tensor_arena[tensor_arena_size] __attribute__((aligned(16)));

    printf("Tensor_arena allocated successfully\n");

    // Allocate Tensor Arena
    PersistentAllocator allocator;
    PersistentAllocator_Init(&allocator, tensor_arena, tensor_arena_size);

    printf("Persistent allocator initiated successfully\n");

    size_t input1_tensor_size = ELEMENTWISE_ADD_INPUT1_TENSOR_SIZE;
    size_t input2_tensor_size = ELEMENTWISE_ADD_INPUT2_TENSOR_SIZE;
    size_t output_tensor_size = ELEMENTWISE_ADD_OUTPUT_TENSOR_SIZE;





    printf("Before allocating input1_tensor\n");
    printf("allocator bufferhead: %p\n", PersistentAllocator_GetBufferHead(&allocator));
    printf("allocator tail: %p\n", PersistentAllocator_GetTailTemp(&allocator));
    // Allocate for input1_tensor
    int8_t* input1_tensor = (int8_t*)PersistentAllocator_Allocate(&allocator, input1_tensor_size, 16);
    if (input1_tensor) {
        for (int i = 0; i < input1_tensor_size; i++) {
        input1_tensor[i] = 5;  // Writing input tensors to SRAM
        }
    }


    printf("Before allocating input2_tensor\n");
    printf("allocator bufferhead: %p\n", PersistentAllocator_GetBufferHead(&allocator));
    printf("allocator tail: %p\n", PersistentAllocator_GetTailTemp((&allocator)));

    // Allocate for input2_tensor
    int8_t* input2_tensor = (int8_t*)PersistentAllocator_Allocate(&allocator, input2_tensor_size, 16);
    if (input2_tensor) {
        for (int i = 0; i < input2_tensor_size; i++) {
        input2_tensor[i] = 3;  // Writing input tensors to SRAM
        }
    }



    printf("Before allocating output_tensor\n");
    printf("allocator bufferhead: %p\n", PersistentAllocator_GetBufferHead(&allocator));
    printf("allocator tail: %p\n", PersistentAllocator_GetTailTemp((&allocator)));
    // Allocate for output_tensor (dont have to initiallize anything to it, but we do zero for safety?)
    int8_t* output_tensor = (int8_t*)PersistentAllocator_Allocate(&allocator, output_tensor_size, 16);
    if (output_tensor) {
        for (int i = 0; i < output_tensor_size; i++) {
        output_tensor[i] = 0;
        }
    }



    // print values
    printf("BEFORE INVOKE\n");
    PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
    PrintTensor("input1_tensor", input1_tensor, ELEMENTWISE_ADD_INPUT1_TENSOR_SIZE);
    PrintTensor("input2_tensor", input2_tensor, ELEMENTWISE_ADD_INPUT2_TENSOR_SIZE);
    PrintTensor("output_tensor", output_tensor, ELEMENTWISE_ADD_OUTPUT_TENSOR_SIZE);



    // No model weights for ELEMENTWISE OP
    //uint8_t model_weight_tensor[0] __attribute__((aligned(16)));
    uint8_t* model_weight_tensor = NULL;


    const size_t num_tensors = 6;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] =(uint64_t)(uintptr_t)(model_weight_tensor);    //model weights
    base_addrs[1] =(uint64_t)(uintptr_t)(tensor_arena);           //Tensor arena pointer
    base_addrs[2] =(uint64_t)(uintptr_t)(tensor_arena);           //Fast scratch, just keep same as tensor arena for now
    base_addrs[3] =(uint64_t)(uintptr_t)(input1_tensor);           // Input tensor (lies in the tensor arena)
    base_addrs[4] =(uint64_t)(uintptr_t)(input2_tensor);           // Input tensor (lies in the tensor arena)
    base_addrs[5] =(uint64_t)(uintptr_t)(output_tensor);          // Output tensor (lies in the tensor arena)


    base_addrs_size[0] = 0;
    base_addrs_size[1] = tensor_arena_size;
    base_addrs_size[2] = tensor_arena_size;
    base_addrs_size[3] = ELEMENTWISE_ADD_INPUT1_TENSOR_SIZE;
    base_addrs_size[4] = ELEMENTWISE_ADD_INPUT2_TENSOR_SIZE;
    base_addrs_size[5] = ELEMENTWISE_ADD_OUTPUT_TENSOR_SIZE;


    
    

    

    if(run_cms(command_stream, command_stream_size, base_addrs, base_addrs_size, num_tensors) != 0) {
        printf("run_cms call failed\n");
        return -1;
    } else {
        printf("run_cms called successfully\n");
    }




    // print values
    printf("AFTER INVOKE\n");
    printf("tensor_arena\n");
    PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
    PrintTensor("input1_tensor", input1_tensor, ELEMENTWISE_ADD_INPUT1_TENSOR_SIZE);
    PrintTensor("input2_tensor", input2_tensor, ELEMENTWISE_ADD_INPUT2_TENSOR_SIZE);
    PrintTensor("output_tensor", output_tensor, ELEMENTWISE_ADD_OUTPUT_TENSOR_SIZE);

    ////Write the values from the arena tensor to the output tensor
    //for (int i = 0; i < ELEMENTWISE_ADD_OUTPUT_TENSOR_SIZE; i++) {
    //    output[i] = output_tensor[i];  // Writing input tensors to SRAM
    //}

    //printf("input1_tensor\n");
    //printf("Argagnoreg\n");
    //PrintTensor(tmpinput1ptr, ELEMENTWISE_ADD_INPUT1_TENSOR_SIZE);
    //printf("input2_tensor\n");
    //PrintTensor(tmpinput2ptr, ELEMENTWISE_ADD_INPUT2_TENSOR_SIZE);
    //printf("output_tensor\n");
    //PrintTensor(tmpoutputptr, ELEMENTWISE_ADD_OUTPUT_TENSOR_SIZE);




    return 0;


}





int run_npu_op(
    uint8_t* command_stream,
    size_t command_stream_size,

    int8_t* tensor_arena,
    size_t tensor_arena_size,

    int8_t* input_tensor,
    size_t input_tensor_size,

    int8_t* weight_tensor,
    size_t weight_tensor_size,

    int8_t* output_tensor,
    size_t output_tensor_size


) {
     // Assign base addrs
     const size_t num_tensors = 5;
     uint64_t base_addrs[num_tensors];
     size_t base_addrs_size[num_tensors];
 
     base_addrs[0] = (uint64_t)(uintptr_t)weight_tensor;   // Model weights
     base_addrs[1] = (uint64_t)(uintptr_t)tensor_arena;   // Tensor arena pointer
     base_addrs[2] = (uint64_t)(uintptr_t)tensor_arena;   // Fast scratch, same as tensor arena for now
     base_addrs[3] = (uint64_t)(uintptr_t)input_tensor;   // Input tensor (in tensor arena)
     base_addrs[4] = (uint64_t)(uintptr_t)output_tensor;  // Output tensor (in tensor arena)
 
     base_addrs_size[0] = weight_tensor_size;
     base_addrs_size[1] = tensor_arena_size;
     base_addrs_size[2] = tensor_arena_size;
     base_addrs_size[3] = input_tensor_size;
     base_addrs_size[4] = output_tensor_size;
 
 
 
 
     // Run NPU commands
     if(run_cms(command_stream, command_stream_size, base_addrs, base_addrs_size, num_tensors) != 0) {
         printf("run_cms call failed\n");
         return -1;
     }
 
 
     if (DEBUG_MODE) {
         //print tensor values after
         printf("AFTER INVOKE\n");
         PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
         PrintTensor("input_tensor", input_tensor, input_tensor_size);
         PrintTensor("output_tensor", output_tensor, output_tensor_size);
         //PrintTensor("weights_tensor_on_sram", weight_tensor_on_sram, weight_size);
         PrintTensor("weight_tensor", weight_tensor, weight_tensor_size);
     }



     return 0;

}






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
)
{



    int8_t tensor_arena[tensor_arena_size] __attribute__((aligned(16)));



    // Allocate Tensor Arena
    // Initialize the allocator
    PersistentAllocator allocator;
    PersistentAllocator_Init(&allocator, tensor_arena, tensor_arena_size);


    // Manually set the relative addresses
    int8_t* input1_tensor = PersistentAllocator_GetAbsPointer(&allocator, 
        0x00);
    int8_t* input2_tensor = PersistentAllocator_GetAbsPointer(&allocator, 
        0x10);
    int8_t* output_tensor = PersistentAllocator_GetAbsPointer(&allocator, 
        0x00);

    
    // Fill in input values to the tensor arena
    for (size_t i = 0; i < input1_tensor_size; i++) {
        input1_tensor[i] = input1[i];
    }

    for (size_t i = 0; i < input2_tensor_size; i++) {
        input2_tensor[i] = input2[i];
    }





    if (DEBUG_MODE) {

        // print values
        printf("BEFORE INVOKE\n");
        PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
        PrintTensor("input1_tensor", input1_tensor, input1_tensor_size);
        PrintTensor("input2_tensor", input2_tensor, input2_tensor_size);
        PrintTensor("output_tensor", output_tensor, output_tensor_size);
    }



    // Assign base addrs
    const size_t num_tensors = 6;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] = (uint64_t)(intptr_t)scales_tensor;   // Model weights
    base_addrs[1] = (uint64_t)(intptr_t)tensor_arena;   // Tensor arena pointer
    base_addrs[2] = (uint64_t)(intptr_t)tensor_arena;   // Fast scratch, same as tensor arena for now
    base_addrs[3] = (uint64_t)(intptr_t)input1_tensor;   // Input tensor (in tensor arena)
    base_addrs[4] = (uint64_t)(intptr_t)input2_tensor;   // Input tensor (in tensor arena)
    base_addrs[5] = (uint64_t)(intptr_t)output_tensor;  // Output tensor (in tensor arena)

    base_addrs_size[0] = scales_tensor_size;
    base_addrs_size[1] = tensor_arena_size;
    base_addrs_size[2] = tensor_arena_size;
    base_addrs_size[3] = input1_tensor_size;
    base_addrs_size[4] = input2_tensor_size;
    base_addrs_size[5] = output_tensor_size;




    //set params
    //const uint8_t set_weight_len[40] __attribute__((aligned(16))) = {
    //    0x43, 0x4f, 0x50, 0x31, 0x01, 0x00, 0x10, 0x00, 0x08, 0x30, 0x00, 0x00, 0x00, 0x00, 0x06, 0x10, 0x05, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x02, 0x00, 0x08, 0x00, 
    //    0x21, 0x00, 0x00, 0x00,
    //    0x90, 0x00, 0x00, 0x00};
//
    //if(run_cms(set_weight_len, 40, base_addrs, base_addrs_size, num_tensors) != 0) {
    //    printf("run_cms call failed\n");
    //    return -1;
    //}
        
    


    // Run NPU commands
    if(run_cms(command_stream, command_stream_size, base_addrs, base_addrs_size, num_tensors) != 0) {
        printf("run_cms call failed\n");
        return -1;
    }


    if (DEBUG_MODE) {
        //print tensor values after
        printf("AFTER INVOKE\n");
        PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
        PrintTensor("input1_tensor", input1_tensor, input1_tensor_size);
        PrintTensor("input2_tensor", input2_tensor, input2_tensor_size);
        PrintTensor("output_tensor", output_tensor, output_tensor_size);
    }

 
    //Write result to output
    printf("writing output to output_tensor:\n");
    for (size_t i = 0; i < output_tensor_size; i++) {
        output[i] = output_tensor[i];
        printf(" %f", output_tensor[i]);
    }printf("\n");
    


    return 0;




}




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
)
{



    int8_t tensor_arena[tensor_arena_size] __attribute__((aligned(16)));



    // Allocate Tensor Arena
    // Initialize the allocator
    PersistentAllocator allocator;
    PersistentAllocator_Init(&allocator, tensor_arena, tensor_arena_size);


    // Manually set the relative addresses
    int8_t* in_spk_tensor = PersistentAllocator_GetAbsPointer(&allocator, 
        0x00);
    int8_t* v_mem_tensor = PersistentAllocator_GetAbsPointer(&allocator, 
        0x20);
    int8_t* decay_tensor = PersistentAllocator_GetAbsPointer(&allocator, 
        0x40);

    int8_t* v_mem_new_tensor = PersistentAllocator_GetAbsPointer(&allocator, 
        0x40);
    int8_t* out_spk_tensor = PersistentAllocator_GetAbsPointer(&allocator, 
        0x20);

    
    // Fill in input values to the tensor arena
    for (size_t i = 0; i < in_spk_tensor_size; i++) {
        in_spk_tensor[i] = in_spk[i];
    }

    for (size_t i = 0; i < v_mem_tensor_size; i++) {
        v_mem_tensor[i] = v_mem[i];
    }

    for (size_t i = 0; i < decay_tensor_size; i++) {
        decay_tensor[i] = decay[i];
    }






    if (DEBUG_MODE) {

        // print values
        printf("BEFORE INVOKE\n");
        PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
        PrintTensor("in_spk_tensor", in_spk_tensor, in_spk_tensor_size);
        PrintTensor("v_mem_tensor", v_mem_new_tensor, v_mem_new_tensor_size);
        PrintTensor("decay_tensor", decay_tensor, decay_tensor_size);
        PrintTensor("v_mem_new_tensor", v_mem_new_tensor, v_mem_new_tensor_size);
        PrintTensor("out_spk_tensor", out_spk_tensor, out_spk_tensor_size);
    }



    // Assign base addrs
    const size_t num_tensors = 6;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] = (uint64_t)(intptr_t)weight_tensor;   // Model weights
    base_addrs[1] = (uint64_t)(intptr_t)tensor_arena;   // Tensor arena pointer
    base_addrs[2] = (uint64_t)(intptr_t)tensor_arena;   // Fast scratch, same as tensor arena for now
    base_addrs[3] = (uint64_t)(intptr_t)in_spk_tensor;   // Input tensor (in tensor arena)
    base_addrs[4] = (uint64_t)(intptr_t)v_mem_tensor;   // Input tensor (in tensor arena)
    base_addrs[5] = (uint64_t)(intptr_t)decay_tensor;
    base_addrs[6] = (uint64_t)(intptr_t)v_mem_new_tensor;
    base_addrs[7] = (uint64_t)(intptr_t)out_spk_tensor;  

    base_addrs_size[0] = weight_tensor_size;
    base_addrs_size[1] = tensor_arena_size;
    base_addrs_size[2] = tensor_arena_size;
    base_addrs_size[3] = in_spk_tensor_size;
    base_addrs_size[4] = v_mem_tensor_size;
    base_addrs_size[5] = decay_tensor_size;
    base_addrs_size[6] = v_mem_new_tensor_size;
    base_addrs_size[7] = out_spk_tensor_size;




    // Run NPU commands
    if(run_cms(command_stream, command_stream_size, base_addrs, base_addrs_size, num_tensors) != 0) {
        printf("run_cms call failed\n");
        return -1;
    }


    if (DEBUG_MODE) {
        //print tensor values after
        printf("AFTER INVOKE\n");
        PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
        PrintTensor("in_spk_tensor", in_spk_tensor, in_spk_tensor_size);
        PrintTensor("v_mem_tensor", v_mem_new_tensor, v_mem_new_tensor_size);
        PrintTensor("decay_tensor", decay_tensor, decay_tensor_size);
        PrintTensor("v_mem_new_tensor", v_mem_new_tensor, v_mem_new_tensor_size);
        PrintTensor("out_spk_tensor", out_spk_tensor, out_spk_tensor_size);
    }

 
    //Write result to output
    printf("writing output to output_tensor:\n");
    for (size_t i = 0; i < v_mem_new_tensor_size; i++) {
        v_mem_new[i] = v_mem_new_tensor[i];
    }

    for (size_t i = 0; i < out_spk_tensor_size; i++) {
        out_spk[i] = out_spk_tensor[i];
    }


    return 0;




}


*/

















/*



int matmul(uint8_t* input, uint8_t* output)
{

    //Get length of command stream
    const uint8_t* command_stream = GetMatMulCMSPointer();
    size_t command_stream_size = GetMatMulCMSLen();

    uint8_t tensor_arena[MATMUL_TENSOR_ARENA_SIZE] __attribute__((aligned(16)));



    // Allocate Tensor Arena
    // Initialize the allocator
    PersistentAllocator allocator;
    PersistentAllocator_Init(&allocator, tensor_arena, MATMUL_TENSOR_ARENA_SIZE);

    // Allocate for input tensor
    uint8_t* input_tensor = (uint8_t*)PersistentAllocator_Allocate(&allocator, 
                                        MATMUL_INPUT_TENSOR_SIZE * sizeof(uint8_t), 
                                        MEM_ALIGNMENT);

    if (input_tensor) {
        for (int i = 0; i < MATMUL_INPUT_TENSOR_SIZE; i++) {
            input_tensor[i] = input[i];  // Writing float values
        }
    }



    // Allocate for output tensor
    uint8_t* output_tensor = (uint8_t*)PersistentAllocator_Allocate(&allocator, 
        MATMUL_OUTPUT_TENSOR_SIZE * sizeof(uint8_t), 
        MEM_ALIGNMENT);

    if (output_tensor) {
        for (int i = 0; i < MATMUL_OUTPUT_TENSOR_SIZE; i++) {
            output_tensor[i] = output[i];  // Writing float values
        }
    }



    //Get weight tensor and allocate for them
    const uint8_t* weight_tensor = GetMatMulWeightsPointer();
    size_t weight_size = GetMatMulWeightsLen();

    // Allocate for output tensor
    uint8_t* weight_tensor_on_sram = (uint8_t*)PersistentAllocator_Allocate(&allocator, 
        weight_size * sizeof(uint8_t), 
        MEM_ALIGNMENT);

    if (weight_tensor_on_sram) {
        for (int i = 0; i < weight_size; i++) {
            weight_tensor_on_sram[i] = 0;  // Init to 0
        }
    }



    if (DEBUG_MODE) {

        // print values
        printf("BEFORE INVOKE\n");
        PrintTensor("tensor_arena", tensor_arena, MATMUL_TENSOR_ARENA_SIZE);
        PrintTensor("input_tensor", input_tensor, MATMUL_INPUT_TENSOR_SIZE);
        PrintTensor("output_tensor", output_tensor, MATMUL_OUTPUT_TENSOR_SIZE);
        PrintTensor("weights_tensor_on_sram", weight_tensor_on_sram, weight_size);
        PrintTensor("weight_tensor", weight_tensor, weight_size);
    }



    // Assign base addrs
    const size_t num_tensors = 5;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] = (uint64_t)(uintptr_t)weight_tensor;   // Model weights
    base_addrs[1] = (uint64_t)(uintptr_t)tensor_arena;   // Tensor arena pointer
    base_addrs[2] = (uint64_t)(uintptr_t)tensor_arena;   // Fast scratch, same as tensor arena for now
    base_addrs[3] = (uint64_t)(uintptr_t)input_tensor;   // Input tensor (in tensor arena)
    base_addrs[4] = (uint64_t)(uintptr_t)output_tensor;  // Output tensor (in tensor arena)

    base_addrs_size[0] = weight_size;
    base_addrs_size[1] = MATMUL_TENSOR_ARENA_SIZE;
    base_addrs_size[2] = MATMUL_TENSOR_ARENA_SIZE;
    base_addrs_size[3] = MATMUL_INPUT_TENSOR_SIZE;
    base_addrs_size[4] = MATMUL_OUTPUT_TENSOR_SIZE;




    // Run NPU commands
    if(run_cms(command_stream, command_stream_size, base_addrs, base_addrs_size, num_tensors) != 0) {
        printf("run_cms call failed\n");
        return -1;
    }


    if (DEBUG_MODE) {
        //print tensor values after
        printf("AFTER INVOKE\n");
        PrintTensor("tensor_arena", tensor_arena, MATMUL_TENSOR_ARENA_SIZE);
        PrintTensor("input_tensor", input_tensor, MATMUL_INPUT_TENSOR_SIZE);
        PrintTensor("output_tensor", output_tensor, MATMUL_OUTPUT_TENSOR_SIZE);
        PrintTensor("weights_tensor_on_sram", weight_tensor_on_sram, weight_size);
        PrintTensor("weight_tensor", weight_tensor, weight_size);
    }

 
    //Write result to output
    for (size_t i = 0; i < MATMUL_OUTPUT_TENSOR_SIZE; i++) {
        output[i] = output_tensor[i];
    }
    


    return 0;
}



















int matmul_vela(uint8_t* input, uint8_t* output)
{

    //Get length of command stream
    const uint8_t* command_stream = GetMatMulVelaCMSPointer();
    size_t command_stream_size = GetMatMulVelaCMSLen();

    uint8_t tensor_arena[MATMUL_VELA_TENSOR_ARENA_SIZE] __attribute__((aligned(16)));
    // Allocate Tensor Arena
    PersistentAllocator allocator(tensor_arena, MATMUL_VELA_TENSOR_ARENA_SIZE);

    // Allocate for input tensor
    uint8_t* input_tensor = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(MATMUL_VELA_INPUT_TENSOR_SIZE*sizeof(uint8_t), MEM_ALIGNMENT));
    if (input_tensor) {
        for (int i = 0; i < MATMUL_VELA_INPUT_TENSOR_SIZE; i++) {
          input_tensor[i] = input[i];  // Writing float values
        }
    }


    // Allocate for output tensor
    uint8_t* output_tensor = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(MATMUL_VELA_OUTPUT_TENSOR_SIZE*sizeof(uint8_t), MEM_ALIGNMENT));
    if (output_tensor) {
        for (int i = 0; i < MATMUL_VELA_OUTPUT_TENSOR_SIZE; i++) {
          output_tensor[i] = 0;  // Writing float values
        }
    }


    //Get weight tensor and allocate for them
    const uint8_t* weight_tensor = GetMatMulVelaWeightsPointer();
    size_t weight_size = GetMatMulVelaWeightsLen();


    // Allocate for weights & biases
    uint8_t* weight_tensor_on_sram = static_cast<uint8_t*>(allocator.AllocatePersistentBuffer(weight_size*sizeof(uint8_t), MEM_ALIGNMENT));
    if (weight_tensor_on_sram) {
        for (int i = 0; i < weight_size; i++) {
            weight_tensor_on_sram[i] = 0;  // init to 0
        }
    }




     // print values
     printf("BEFORE INVOKE\n");
     PrintTensor("tensor_arena", tensor_arena, MATMUL_VELA_TENSOR_ARENA_SIZE);
     PrintTensor("input_tensor", input_tensor, MATMUL_VELA_INPUT_TENSOR_SIZE);
     PrintTensor("output_tensor", output_tensor, MATMUL_VELA_OUTPUT_TENSOR_SIZE);
     PrintTensor("weights_tensor_on_sram", weight_tensor_on_sram, weight_size);
     PrintTensor("weight_tensor", weight_tensor, weight_size);
     



    // Assign base addrs
    const size_t num_tensors = 5;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(weight_tensor));    //model weights
    base_addrs[1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));           //Tensor arena pointer
    base_addrs[2] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_arena));           //Fast scratch, just keep same as tensor arena for now
    base_addrs[3] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_tensor));           // Input tensor (lies in the tensor arena)
    base_addrs[4] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(output_tensor));          // Output tensor (lies in the tensor arena)

    base_addrs_size[0] = weight_size;
    base_addrs_size[1] = MATMUL_VELA_TENSOR_ARENA_SIZE;
    base_addrs_size[2] = MATMUL_VELA_TENSOR_ARENA_SIZE;
    base_addrs_size[3] = MATMUL_VELA_INPUT_TENSOR_SIZE;
    base_addrs_size[4] = MATMUL_VELA_OUTPUT_TENSOR_SIZE;




    // Run NPU commands
    if(run_cms(command_stream, command_stream_size, base_addrs, base_addrs_size, num_tensors) != 0) {
        printf("run_cms call failed\n");
        return -1;
    } else {
        printf("run_cms called successfully\n");
    }


    //print tensor values after
    printf("AFTER INVOKE\n");
    PrintTensor("tensor_arena", tensor_arena, MATMUL_VELA_TENSOR_ARENA_SIZE);
    PrintTensor("input_tensor", input_tensor, MATMUL_VELA_INPUT_TENSOR_SIZE);
    PrintTensor("output_tensor", output_tensor, MATMUL_VELA_OUTPUT_TENSOR_SIZE);
    PrintTensor("weights_tensor_on_sram", weight_tensor_on_sram, weight_size);
    PrintTensor("weight_tensor", weight_tensor, weight_size);

 
    //Write result to output
    for (size_t i = 0; i < MATMUL_VELA_OUTPUT_TENSOR_SIZE; i++) {
        output[i] = output_tensor[i];
    }
    


    return 0;


}

*/


