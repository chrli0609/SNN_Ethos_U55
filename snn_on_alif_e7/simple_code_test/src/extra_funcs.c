#include "include/extra_funcs.h"



#include <stdio.h>



// Function to print uint8_t tensor values
void PrintTensor(const char* tensor_name, const uint8_t* tensor, size_t num_elements) {
    if (!tensor) {
      printf("Tensor is NULL!\n");
      return;
    }

    printf("%s\n", tensor_name);
    for (size_t i = 0; i < num_elements; i++) {
      printf("%u ", tensor[i]);
    }
    printf("\n");
}
