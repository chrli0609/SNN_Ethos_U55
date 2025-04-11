#pragma once



#include <stddef.h>
#include <stdint.h>






#define MATMUL_TENSOR_ARENA_SIZE 640
#define MATMUL_INPUT_TENSOR_SIZE 16
#define MATMUL_OUTPUT_TENSOR_SIZE 32




const uint8_t * GetMatMulCMSPointer();

size_t GetMatMulCMSLen();


const uint8_t * GetMatMulWeightsPointer();

size_t GetMatMulWeightsLen();

