#pragma once


#include <stdint.h>
#include <stddef.h>



void quantize_array_float_to_int16(
    const float *input,     // Input float array
    int16_t *output,        // Output int16 array
    int length,             // Number of elements
    float scale,            // Quantization scale
    int32_t zero_point      // Quantization zero-point
  );
void dequantize_array_int16_to_float(
    const int16_t *input,   // Input int16 array
    float *output,          // Output float array
    int length,             // Number of elements
    float scale,            // Quantization scale
    int32_t zero_point      // Quantization zero-point
  );


void PrintInt16Tensor(const char* tensor_name, const int16_t* tensor, size_t num_elements);
void PrintInt8Tensor(const char* tensor_name, const int8_t* tensor, size_t num_elements);
void PrintFloatTensor(const char* tensor_name, const float* tensor, size_t num_elements);