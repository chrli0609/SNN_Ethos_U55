#include <stdint.h>
#include <stddef.h>


#include <stdio.h>
#include <string.h>


#include <cstdlib> //malloc

#include "ethosu_driver.h"



//#include "conv2d_model.hpp"



int create_tensors(uint8_t*** base_addrs, size_t** base_addrs_size, int* num_tensors) {
    
    size_t n = 3;
    //size_t m = 128;

    int m [] = {
        8*8*16,     //input tensor
        16*2*2*16,  //weight tensor
        4*4*16      //output tensor
    };


    uint8_t init_vals [] = {
        4,
        1,
        0
    };




    // Check that tensor sizes are divisible by 16
    for (int i = 0; i < n; i++) {
        if (m[i] % 16 != 0) {
            printf("ERROR: tensors sizes not divisible by 16, is instead: %d\n", m[i]%16);
            return -1;
        }
    }


    



    *num_tensors = n;

    // Ensure m is a multple of 16 for aligned_alloc
    //size_t aligned_size = (m + 15) & ~15;

    // Allocate base address and size arrays
    *base_addrs = (uint8_t **)malloc(3 * sizeof(uint64_t*));
    *base_addrs_size = (size_t *)malloc(3 * sizeof(size_t));

    if (*base_addrs == NULL || *base_addrs_size == NULL) {
        perror("Memory allocation failed for base_addrs or base_addrs_size");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < n; i++) {

        (*base_addrs)[i] = (uint8_t*)aligned_alloc(16,  m[i]);

        if ((*base_addrs)[i] == NULL) {
            perror("Failed to allocate aligned memory");
            exit(EXIT_FAILURE);
        }
        

        // store tensor size
        (*base_addrs_size)[i] = m[i];
        printf("base_addrs[%zu] address: %p\n", i, (void *)(*base_addrs[i]));

        // Initialize values to zero
        memset((*base_addrs)[i], init_vals[i], m[i]);

    }

    return 0;
    

}


void free_tensors(uint8_t** base_addrs, size_t* base_addrs_size, int num_tensors)
{

    for (size_t i = 0; i < num_tensors; i++) {
        free(base_addrs[i]);
    }
    free(base_addrs);
    free(base_addrs_size);

}

void print_uint8_array(uint8_t* array, size_t size) {
    printf("Array contents: ");
    for (size_t i = 0; i < size; i++) {
        printf("%u ", array[i]);  // Print as unsigned integer
    }
    printf("\n");
}


int create_n_run_cmd_stream() {

    //Get length of command stream
    size_t command_stream_size = GetModelLen();
    
    //create input arrays
    uint8_t** base_addrs;
    size_t* base_addrs_size;
    int num_tensors;

    //allocate memory for them
    if (create_tensors(&base_addrs, &base_addrs_size, &num_tensors) == -1) {
        printf("Failed to allocate memory for tensors\n");
        return -1;
    }


    printf("tensor values before inference:\n");    
    printf("input_data size: %d\n", base_addrs_size[0]);
    print_uint8_array(base_addrs[0], base_addrs_size[0]);

    printf("weight_data_size: %d\n", base_addrs_size[1]);
    print_uint8_array(base_addrs[1], base_addrs_size[1]);

    printf("output_data_size: %d\n", base_addrs_size[2]);
    print_uint8_array(base_addrs[2], base_addrs_size[2]);
    


    // Reserve the Ethos-U driver
    struct ethosu_driver*const drv = ethosu_reserve_driver();
    if (!drv) {
        printf("Failed to reserve Ethos-U driver\n");
        return -1;
    }

    printf("Before invoke_v3: &drv = %d, drv = %d\n", (void*)&drv, (void*)drv);


    //test_npu_driver(drv, command_stream, command_stream_size, 
    //    base_addrs, base_addrs_size, num_tensors, NULL);

    int result = ethosu_invoke_v3(drv, command_stream, command_stream_size, 
        (uint64_t*)base_addrs, base_addrs_size, num_tensors, NULL);


    //struct ethosu_driver*const new_drv = (struct ethosu_driver*)536879144;

    printf("After invoke_v3: &drv = %d, drv = %d\n", (void*)&drv, (void*)drv);



    printf("result: %d\n", result);
    if (result == -1) {
        printf("ERRORRRRRR\n");
        return -1;
    } else {
        printf("its all okk\n");
    }
    


    ethosu_release_driver(drv);

    printf("Driver release successfully\n");


    // Print results
    size_t input_size = sizeof(base_addrs[0]) / sizeof((base_addrs[0])[0]);
    size_t weight_size = sizeof(base_addrs[1]) / sizeof((base_addrs[1])[0]);
    size_t output_size = sizeof(base_addrs[2]) / sizeof((base_addrs[2])[0]);


    printf("input_data size: %d\n", input_size);
    printf("input_data size: %d\n", base_addrs_size[0]);
    print_uint8_array(base_addrs[0], base_addrs_size[0]);

    printf("weight_data_size: %d\n", base_addrs_size[1]);
    print_uint8_array(base_addrs[1], base_addrs_size[1]);

    printf("output_data_size: %d\n", base_addrs_size[2]);
    print_uint8_array(base_addrs[2], base_addrs_size[2]);
    


    //free the tensors
    free_tensors(base_addrs, base_addrs_size, num_tensors);


    printf("Tensors freed successfully\n");

  
    return 0;

}
