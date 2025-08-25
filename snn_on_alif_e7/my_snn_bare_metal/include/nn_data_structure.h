#pragma once

#include <stdint.h> //int8_t
#include <stddef.h> //size_t



#define MEM_ALIGNMENT 16

#define TENSOR_INIT_VALUE 0
#define MAX_NUM_PRINTED_VALUES_PER_LINE 10

//// Simple Persistent Buffer Allocator (like PersistentArenaBufferAllocator)
//typedef struct {
    //int8_t* buffer_head;  // Start of buffer
    //int8_t* tail_temp;    // Current tail position
//} PersistentAllocator;



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
    //int8_t* input;
    //size_t input_size;

    //int8_t* output;
    //size_t output_size;

    //int8_t* update_nxt;
    //int8_t* update_curr;


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



void Tensor_Print(Tensor* tensor);
void Tensor_Print_Quant_Values(Tensor* tensor);


Tensor* NNLayer_Get_Tensor(NNLayer* nnlayer, const char* tensor_name);



void* aligned_malloc(size_t required_bytes, size_t alignment);
void aligned_free(void* p);

//void PersistentAllocator_Init(PersistentAllocator* allocator, int8_t* arena, size_t size);
//void* PersistentAllocator_GetAbsPointer(PersistentAllocator* allocator, size_t relative_addr);
//void* PersistentAllocator_Allocate(PersistentAllocator* allocator, size_t size, size_t alignment);


//int8_t* PersistentAllocator_GetBufferHead(PersistentAllocator* allocator);
//int8_t* PersistentAllocator_GetTailTemp(PersistentAllocator* allocator);

NNLayer* NNLayer_Init(size_t num_tensors, size_t num_regions);
//int NNLayer_Assign(NNLayer* layer, size_t element, int8_t* tensor_ptr, size_t tensor_size, 
                   //float scale, int zero_point, const char* tensor_name);
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


    //float in_spk_scale,
    //int in_spk_zero_point,

    //float v_mem_scale,
    //int v_mem_zero_point,
    //float time_not_updated_scale,
    //int time_not_updated_zero_point,

    //float out_spk_scale,
    //int out_spk_zero_point
    int is_last_layer);

void NNLayer_Free(NNLayer* layer);
void NNLayer_DequantizeAndPrint(const NNLayer* layer);


NN_Model* NN_Model_Init(int8_t* total_arena_tensor, NNLayer* first_nnlayer, size_t input_size, size_t output_size, size_t num_time_steps, size_t out_spk_sum_tensor_idx);








//typedef struct {

    //// Const tensors
    //float* weights;
    //float* biases;
    //float* vth;
    //float* beta;

    //// Non const tensors
    //float* v_mem;
    //float* time_since_last_update;



    //float* input;
    //size_t input_size;

    //float* output;
    //size_t output_size;
    //float* out_spk_sum;

    //struct NNLayer_CPU* next_layer;

//} NNLayer_CPU;



//typedef struct {
    //size_t input_size;
    //size_t output_size;


    //// Num time steps to process each input sample
    //size_t num_time_steps;

    //size_t num_layers;


    //NNLayer_CPU* first_nnlayer;
    //NNLayer_CPU* last_nnlayer;

    //float* input;
    //float* output;

    //float* out_spk_sum;

//}NN_Model_CPU;