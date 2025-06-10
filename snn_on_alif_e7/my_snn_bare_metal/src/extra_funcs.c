#include "include/extra_funcs.h"



#include <stdio.h>
#include <math.h>


#include "pm.h"                 //SystemCoreClock


size_t arg_max(int8_t* array, size_t array_len, float scale, int zero_point) {
    // Get the max value
    size_t max_value = 0;
    size_t max_spk_idx = 0;
    size_t neuron_sum = 0;

    for (size_t i = 0; i < array_len; i++) {
        neuron_sum = (size_t)(array[i] - zero_point) * scale;
        if (neuron_sum > max_value) {
            max_value = neuron_sum;
            max_spk_idx = i;
        }
    }
    //printf("Prediction: %d\n", max_spk_idx);

    return max_spk_idx;

}


uint32_t volatile ms_ticks = 0;
void SysTick_Handler (void) {
    ms_ticks++;
}
uint32_t debug_start_timer(void) {
    return ms_ticks;
}

uint32_t debug_end_timer(uint32_t start_tick) {
    
    //Current time - the time we started --> time elapsed
    uint32_t elapsed_ticks = ms_ticks - start_tick;
    return elapsed_ticks;

}
void delay(uint32_t nticks){
    uint32_t c_ticks;

    c_ticks = ms_ticks;
    while((ms_ticks - c_ticks) < nticks) __WFE() ;
}



uint32_t start_timer() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    
    uint32_t start = DWT->CYCCNT;

    return start;
    
}
float end_timer(uint32_t start) {

    uint32_t end = DWT->CYCCNT;
    uint32_t elapsed_cycles = end - start;
    float elapsed_ms = (float)elapsed_cycles / (SystemCoreClock / 1000.0f);


    return elapsed_ms;
}




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




// Quantize an float scalar to int8_t scalar
void quantize_float_scalar_to_int8_scalar(
  const float input_val,  // Input float scalar
  int8_t *output,        // Output int8 array
  float scale_reciprocal,            // Reciprocal of Quantization scale (use reciprocal to avoid division)
  int32_t zero_point      // Quantization zero-point
)
{

  int32_t quantized = (int32_t)roundf(input_val * scale_reciprocal) + zero_point;

  // Clamp to [-128, 127]
  if (quantized > 127) quantized = 127;
  if (quantized < -128)   quantized = -128;

  *output = (int8_t)quantized;
}


// Quantize an array of floats to int8_t
void quantize_float_scalar_to_int8_array(
  const float input_val,  // Input float scalar
  int8_t *output,        // Output int8 array
  int length,             // Number of elements
  float scale,            // Quantization scale
  int32_t zero_point      // Quantization zero-point
)
{

  int32_t quantized = (int32_t)roundf(input_val / scale) + zero_point;

  // Clamp to [-128, 127]
  if (quantized > 127) quantized = 127;
  if (quantized < -128)   quantized = -128;

  for (int i = 0; i < length; i++)
  {
    output[i] = (int8_t)quantized;
  }
}



//not in use (?)
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