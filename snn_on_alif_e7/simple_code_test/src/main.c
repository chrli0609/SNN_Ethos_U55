
#include <stdio.h>


#include "BoardInit.h"
//#include "gpio_wrapper.h"    /* GPIO wrapper for LED control */
//#include "Driver_GPIO.h"     /* GPIO driver for LED control */
//#include "GpioSignal.hpp"




#include "ethosu_driver.h"





#include "include/nn_ops.h"
#include "include/lif_model.h"
#include "include/extra_funcs.h"




#include "include/elementwise_mul.h"
//#include "nn_ops/membrane_update_python.h"
#include "include/my_mem_u.h"



const int DEBUG_MODE = 1;



//#include <unistd.h> // notice this! you need it!


#define REGISTER_ADDRESS    0x20020200
#define REGISTER            (*(volatile uint8_t*)REGISTER_ADDRESS)


//void write_to_register(uint8_t value) {
//    REGISTER = value;
//}




//#define _GET_DRIVER_REF(ref, peri, chan) \
//    extern ARM_DRIVER_##peri Driver_##peri##chan; \
//    static ARM_DRIVER_##peri * ref = &Driver_##peri##chan;
//#define GET_DRIVER_REF(ref, peri, chan) _GET_DRIVER_REF(ref, peri, chan)


// Update to use the GPIO associated with P15_1 (W2)
//GET_DRIVER_REF(gpio_led, GPIO, 15);
//GET_DRIVER_REF(gpio_b, GPIO, BOARD_LEDRGB0_B_GPIO_PORT);
//GET_DRIVER_REF(gpio_r, GPIO, BOARD_LEDRGB0_R_GPIO_PORT);

//#define LED_PIN_NO  1









int main() {


    // Initialize the led
    //gpio_b->Initialize(BOARD_LEDRGB0_B_PIN_NO, NULL);
    //gpio_b->PowerControl(BOARD_LEDRGB0_B_PIN_NO, ARM_POWER_FULL);
    //gpio_b->SetDirection(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
    //gpio_b->SetValue(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);

    //gpio_r->Initialize(BOARD_LEDRGB0_R_PIN_NO, NULL);
    //gpio_r->PowerControl(BOARD_LEDRGB0_R_PIN_NO, ARM_POWER_FULL);
    //gpio_r->SetDirection(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
    //gpio_r->SetValue(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);


    //gpio_led->Initialize(LED_PIN_NO, NULL);
    //gpio_led->PowerControl(LED_PIN_NO, ARM_POWER_FULL);
    //gpio_led->SetDirection(LED_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);


    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

    //while (1) {

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
            //MY_MEM_U_IN_SPK_ADDR,
            //MY_MEM_U_IN_CURR_ADDR,

            out_spk
        );


        // expected v_mem = v_mem * decay + in_spk x weights = 0.5 * 0.9 + 16 * (0.8 * 0.2) = 0.45 + 2.56 = 3.01
        PrintFloatTensor("v_mem", v_mem, MY_MEM_U_OUTPUT_LAYER_SIZE);
        PrintFloatTensor("out_spk", out_spk, MY_MEM_U_OUTPUT_LAYER_SIZE);


        





        //##############################################################
        // try doing a single layer


        /*
        //create v_mem and initialize values to 0
        #include "include/matmul.h"

        uint8_t v_mem[MATMUL_OUTPUT_TENSOR_SIZE];
        for (size_t i = 0; i < MATMUL_OUTPUT_TENSOR_SIZE; i++) {
            v_mem[i] = 0;
        }


        //create input spike code
        uint8_t in_spk[MATMUL_INPUT_TENSOR_SIZE];
        for (size_t i = 0; i < MATMUL_INPUT_TENSOR_SIZE; i++) {
            in_spk[i] = i % 2;
        }




        for (size_t i = 0; i < 2; i++) {


            PrintTensor("v_mem", v_mem, MATMUL_OUTPUT_TENSOR_SIZE);
            PrintTensor("in_spk", in_spk, MATMUL_INPUT_TENSOR_SIZE);




            float beta = 0.9;
            size_t num_time_steps_since_update = 1;

            float decay = beta*num_time_steps_since_update;



            float threshold = 30;

            
            //Create out_spk
            uint8_t out_spk[MATMUL_OUTPUT_TENSOR_SIZE];


            const uint8_t* weight_ptr = GetMatMulWeightsPointer();

            membrane_update(v_mem, in_spk, weight_ptr, decay, threshold, out_spk);



            PrintTensor("v_mem", v_mem, MATMUL_OUTPUT_TENSOR_SIZE);
            PrintTensor("out_spk", out_spk, MATMUL_OUTPUT_TENSOR_SIZE);



        }
        */


        printf("exited to while loop in main\n");
        



        

        //################################################################

   
        
        //gpio_led->SetValue(LED_PIN_NO, GPIO_PIN_OUTPUT_STATE_TOGGLE)
        //gpio_b->SetValue(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_OUTPUT_STATE_TOGGLE);

        
   
        /*
        //SNN model
        long int time_step = 0;
        while (true) {

            //Input layer
            for 



            time_step++;
        }
        */
        

    //}

}