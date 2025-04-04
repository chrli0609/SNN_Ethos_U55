
#include <stdio.h>




#include "include/lif_model.h"
#include "include/matmul.h"
#include "include/nn_ops.h"
#include "include/extra_funcs.h"





struct myTuple {
    int spike;
    float membrane_voltage;
};





// Inputs:
//  * v_mem, dim=(LAYER i)
//  * in_spk, dim=(layer i-1)
//  * weights, dim=(depends on encoding)
//  * beta, scalar
//  * threshold, scalar
// Output:
//  * v_mem, dim=(layer i)
//  * out_spk, dim=(layer i)
int membrane_update(
    uint8_t* v_mem,
    uint8_t* in_spk,
    const uint8_t* weight_tensor_ptr,

    float decay,
    float threshold,


    uint8_t* out_spk
) {


    //Compute delay
    for (size_t i = 0; i < MATMUL_OUTPUT_TENSOR_SIZE; i++) {
        v_mem[i] = decay * v_mem[i];
    }



    // In_cur = W * X
    uint8_t in_cur[MATMUL_OUTPUT_TENSOR_SIZE];
    if (matmul(in_spk, in_cur) != 0) { return -1; }
    
    PrintTensor("in_cur", in_cur, MATMUL_INPUT_TENSOR_SIZE);



    // Check for spikes & do reset mechanism
    for (size_t i = 0; i < MATMUL_OUTPUT_TENSOR_SIZE; i++) {
        v_mem[i] = v_mem[i] + in_cur[i];

        if (v_mem[i] > threshold) {
            v_mem[i] = v_mem[i] - threshold;
            out_spk[i] = 1;
        
        } else {
            out_spk[i] = 0;
        }
    }


    return 0;

}




struct myTuple leaky_integrate_fire(float membrane_voltage, float x, float w, float beta, float threshold) {

    int spike;
    if (membrane_voltage > threshold) {
        spike = 1;
    } else {
        spike = 0;
    }

    float new_membrane_voltage = beta * membrane_voltage + w * x - spike * threshold;


    struct myTuple r = {spike, new_membrane_voltage};

    return r;
}