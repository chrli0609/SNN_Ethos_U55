#pragma once



#include <stdint.h>
#include <stddef.h>















//int elementwise_add();




int my_mem_u_npu(
    int8_t* tensor_arena,
    size_t tensor_arena_size,


    const uint8_t* command_stream,
    size_t command_stream_size,
    const int8_t* weight_tensor,
    size_t weight_tensor_size,


    const int8_t* exp_lut,
    size_t exp_lut_size

);


