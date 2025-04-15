#include "include/extra_funcs.h"



#include <stdio.h>
#include <math.h>








// Convert float to fixed-point int16_t
int16_t float_to_fixed(float input, int fractional_bits) {
  float scaled = input * (1 << fractional_bits);
  
  // Round to nearest integer
  int32_t temp = (int32_t)(scaled + (scaled >= 0 ? 0.5f : -0.5f));

  // Saturate to int16_t range
  if (temp > 32767) temp = 32767;
  if (temp < -32768) temp = -32768;

  return (int16_t)temp;
}


// Convert fixed-point int16_t to float
float fixed_to_float(int16_t input, int fractional_bits) {
  return ((float)input) / (1 << fractional_bits);
}





// Quantize an array of floats to int8_t
void quantize_array_float_to_int8(
  const float *input,     // Input float array
  int8_t *output,        // Output uint8 array
  int length,             // Number of elements
  float scale,            // Quantization scale
  int32_t zero_point      // Quantization zero-point
)
{
  for (int i = 0; i < length; i++)
  {
      int32_t quantized = (int32_t)roundf(input[i] / scale) + zero_point;

      // Clamp to [0, 255]
      if (quantized > 127) quantized = 127;
      if (quantized < -128)   quantized = -128;

      output[i] = (int8_t)quantized;
  }
}

// Dequantize an array of int8_t to floats
void dequantize_array_int8_to_float(
  const int8_t *input,   // Input uint8 array
  float *output,          // Output float array
  int length,             // Number of elements
  float scale,            // Quantization scale
  int32_t zero_point      // Quantization zero-point
)
{
  for (int i = 0; i < length; i++)
  {
      output[i] = ((int32_t)input[i] - zero_point) * scale;
  }
}





// Function to print int8_t tensor values
void PrintInt16Tensor(const char* tensor_name, const int8_t* tensor, size_t num_elements) {
    if (!tensor) {
      printf("Tensor is NULL!\n");
      return;
    }

    printf("%s\n", tensor_name);
    for (size_t i = 0; i < num_elements; i++) {
      printf("%d ", tensor[i]);
    }
    printf("\n");
}



// Function to print int8_t tensor values
void PrintTensor(const char* tensor_name, const int8_t* tensor, size_t num_elements) {
  if (!tensor) {
    printf("Tensor is NULL!\n");
    return;
  }

  printf("%s\n", tensor_name);
  for (size_t i = 0; i < num_elements; i++) {
    printf("%d ", tensor[i]);
  }
  printf("\n");
}


void PrintFloatTensor(const char* tensor_name, const float* tensor, size_t num_elements) {
  if (!tensor) {
    printf("Tensor is NULL!\n");
    return;
  }

  printf("%s\n", tensor_name);
  for (size_t i = 0; i < num_elements; i++) {
    printf("%f ", tensor[i]);
  }
  printf("\n");
}