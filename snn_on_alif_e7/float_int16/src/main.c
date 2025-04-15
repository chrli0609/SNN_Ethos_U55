
#include <stdio.h>


#include "BoardInit.h"
//#include "gpio_wrapper.h"    /* GPIO wrapper for LED control */
//#include "Driver_GPIO.h"     /* GPIO driver for LED control */
//#include "GpioSignal.hpp"




#include "ethosu_driver.h"


#include "include/nn_ops.h"
#include "include/lif_model.h"
#include "include/extra_funcs.h"


#include "include/my_mem_u.h"



const int DEBUG_MODE = 1;









int main() {


   

    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();


    printf("============ new iteration start =======================\n");




    printf("Test my_mem_u\n");
    float in_spk [MY_MEM_U_INPUT_LAYER_SIZE] = {
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
    };

    float v_mem [MY_MEM_U_OUTPUT_LAYER_SIZE] = {
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    };

    float decay [MY_MEM_U_OUTPUT_LAYER_SIZE] = {
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
    };
    
    float out_spk [MY_MEM_U_OUTPUT_LAYER_SIZE];

    PrintFloatTensor("in_spk", in_spk, MY_MEM_U_INPUT_LAYER_SIZE);
    PrintFloatTensor("v_mem", v_mem, MY_MEM_U_OUTPUT_LAYER_SIZE);
    PrintFloatTensor("decay", decay, MY_MEM_U_OUTPUT_LAYER_SIZE);



    my_mem_update(
        in_spk,
        v_mem,
        decay,

        out_spk
    );

    PrintFloatTensor("v_mem", v_mem, MY_MEM_U_OUTPUT_LAYER_SIZE);
    PrintFloatTensor("out_spk", out_spk, MY_MEM_U_OUTPUT_LAYER_SIZE);





    printf("exited to while loop in main\n");
        



        


}