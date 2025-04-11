#pragma once


#include <stdint.h>
#include <stddef.h>




void quantize_array_float_to_int8(
    const float *input,     // Input float array
    int8_t *output,        // Output uint8 array
    int length,             // Number of elements
    float scale,            // Quantization scale
    int32_t zero_point      // Quantization zero-point
  );


void dequantize_array_int8_to_float(
const int8_t *input,   // Input uint8 array
float *output,          // Output float array
int length,             // Number of elements
float scale,            // Quantization scale
int32_t zero_point      // Quantization zero-point
);

void PrintTensor(const char* tensor_name, const int8_t* tensor, size_t num_elements);
void PrintFloatTensor(const char* tensor_name, const float* tensor, size_t num_elements);