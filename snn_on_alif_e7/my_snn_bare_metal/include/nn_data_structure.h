#pragma once

#include <stdint.h> //int8_t
#include <stddef.h> //size_t



#define MEM_ALIGNMENT 16

#define TENSOR_INIT_VALUE 0
#define MAX_NUM_PRINTED_VALUES_PER_LINE 10



typedef struct {
    int8_t* ptr;
    size_t size;

    size_t region_number;
    size_t relative_addr;

    // Quantization parameters
    float scale;
    int zero_point;
    float scale_reciprocal;

    const char* name;

} Tensor;


typedef struct {
    int8_t* region_start_ptr;
    size_t length;
} MemoryRegion;


typedef struct {

    // Tensors that are always in use
    Tensor* input;
    Tensor* output;
    Tensor* update_nxt_layer;
    Tensor* update_curr_layer;


    float time_of_previous_update;

    // The tensor where most of the memory allocated for the NPU is actually stored
    // (i.e. tmp1, tmp2, v_mem, time_not_updated, update_nxt_layer)
    // Memory Segment should be allocated in connectivity.h

    const uint8_t* command_stream;
    size_t command_stream_length;

    // Memory Regions
    MemoryRegion** memory_regions;
    size_t num_regions;

    // Array of Tensors in layer
    Tensor** tensors;
    size_t num_tensors;

    struct NNLayer* next_layer;

} NNLayer;

typedef struct {

    // Num time steps to process each input sample
    size_t num_time_steps;
    size_t num_layers;


    int8_t* total_tensor_arena;
    NNLayer* first_nnlayer;
    NNLayer* last_nnlayer;

    Tensor* input;
    Tensor* output;
    Tensor* out_spk_sum;

} NN_Model;


// Tensor functions
void Tensor_Print(Tensor* tensor);
void Tensor_Print_Quant_Values(Tensor* tensor);


// NNLayer functions
Tensor* NNLayer_Get_Tensor(NNLayer* nnlayer, const char* tensor_name);


NNLayer* NNLayer_Init(size_t num_tensors, size_t num_regions);

int NNLayer_Assign(
    NNLayer* nnlayer,
    const uint8_t* command_stream,
    size_t command_stream_length,

    int8_t** memory_region_ptrs,
    size_t* memory_region_sizes,
    size_t* memory_region_region_numbers,
    size_t num_regions,

    const char* in_spk,
    size_t input_layer_size,
    const char* out_spk,
    size_t output_layer_size,

    char** tensor_names,
    size_t* tensor_relative_addrs,
    size_t* tensor_regions,
    size_t* tensor_sizes,
    float* tensor_scales,
    int* tensor_zero_points,
    size_t num_tensors,
    int is_last_layer);

void NNLayer_Free(NNLayer* layer);
void NNLayer_DequantizeAndPrint(const NNLayer* layer);






// NN_MODEL functions
NN_Model* NN_Model_Init(int8_t* total_arena_tensor, NNLayer* first_nnlayer, size_t input_size, size_t output_size, size_t num_time_steps);




