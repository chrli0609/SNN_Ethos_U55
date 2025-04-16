#pragma once
#include <stddef.h>
#include <stdint.h>




#define MY_MEM_U_TENSOR_ARENA_SIZE 640
#define MY_MEM_U_INPUT_LAYER_SIZE 16
#define MY_MEM_U_OUTPUT_LAYER_SIZE 32

// Input/output addresses (Relative Addressing)
#define MY_MEM_U_IN_SPK_ADDR 0
#define MY_MEM_U_WEIGHT_ADDR 336
#define MY_MEM_U_BIAS_ADDR 16
#define MY_MEM_U_IN_CURR_ADDR 480
#define MY_MEM_U_V_MEM_ADDR 512
#define MY_MEM_U_DECAY_ADDR 544
#define MY_MEM_U_DECAYED_MEM_ADDR 576
#define MY_MEM_U_WEIGHT_LEN 144
#define MY_MEM_U_BIAS_LEN 320

//Quantization Params
#define MY_MEM_U_IN_SPK_SCALE 0.00392156862745098
#define MY_MEM_U_IN_SPK_ZERO_POINT -128
#define MY_MEM_U_WEIGHT_SCALE 0.009999999999999998
#define MY_MEM_U_WEIGHT_ZERO_POINT 0
#define MY_MEM_U_IN_CURR_SCALE 0.01764705882352941
#define MY_MEM_U_IN_CURR_ZERO_POINT -100
#define MY_MEM_U_V_MEM_SCALE 0.01568627450980392
#define MY_MEM_U_V_MEM_ZERO_POINT -128
#define MY_MEM_U_DECAY_SCALE 0.00392156862745098
#define MY_MEM_U_DECAY_ZERO_POINT -128
#define MY_MEM_U_DECAYED_MEM_SCALE 0.00392156862745098
#define MY_MEM_U_DECAYED_MEM_ZERO_POINT -128






const uint8_t * Getmy_mem_uCMSPointer();


size_t Getmy_mem_uCMSLen();






const int8_t * Getmy_mem_uWeightsPointer();

size_t Getmy_mem_uWeightsLen();


