
#include <stdio.h>

#include <math.h> //for log()


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
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2
        
        };

        printf("0.5");

        float ln_beta [MY_MEM_U_OUTPUT_LAYER_SIZE] = {


            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
            log(0.72),
        };

        printf("1");

        float vth [MY_MEM_U_OUTPUT_LAYER_SIZE] = {
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
            1.3,
        };
        printf("2");
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

        printf("3");
        float time_not_updated [MY_MEM_U_OUTPUT_LAYER_SIZE] = {
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
            5.3,
        };
        printf("4");
        float out_spk [MY_MEM_U_OUTPUT_LAYER_SIZE];

        //PrintFloatTensor("in_spk", in_spk, MY_MEM_U_INPUT_LAYER_SIZE);
        //PrintFloatTensor("v_mem", v_mem, MY_MEM_U_OUTPUT_LAYER_SIZE);
        //PrintFloatTensor("decay", decay, MY_MEM_U_OUTPUT_LAYER_SIZE);


        //bias = 0, scale = 1, shift = 0 --> -112
        //bias = 3, scale = 1, shift = 0 --> -109
        //bias = 10, scale = 1, shift = 0 --> -102
        //bias = 10, scale = 2, shift = 1 --> -102
        //bias = 10, scale = 2**7, shift = 7 --> -102
        //bias = 10, scale = 2, shift = 0 --> -76

        // Set output scale = 1, zero_point = 0

        //bias = 10, scale = 2, shift = 0 --> 52
        //bias = 0, scale = 1, shift = 0 --> 16
        //bias = 0, scale = 2, shift = 0 --> 32
        //bias = 3, scale = 2, shift = 0 --> 38
        //bias = 3, scale = 2, shift = 1 --> 19 (16 + 3)

        // ( sum[(ifm - Z_x) * (w - Z_w)] + bias ) * (scale >> shift) + Z_out

        // Set IFM scale = 2, zero_point = 0

        // Set OFM scale = 0.005, zero_point = -128
        //bias = 3, scale = 1, shift = 0 --> -109   (16 + 3*1 -128)
        //bias = 3, scale = 8, shift = 3 --> -109
        //bias = 3, scale = 8, shift = 2 --> -90    (16 + 3)*2 + (-128))

        // Set OFM scale = 0.01, zero_point = 0

        //bias = 3, scale = 8, shift = 2 --> 38     ((16 + 3)*2 + 0)


            printf("5");


        my_mem_update(
            in_spk,

            ln_beta,
            vth,
            v_mem,
            time_not_updated,

            out_spk
        );


        // expected v_mem = v_mem * decay + in_spk x weights = 0.5 * 0.9 + 16 * (0.8 * 0.2) = 0.45 + 2.56 = 3.01
        //PrintFloatTensor("v_mem", v_mem, MY_MEM_U_OUTPUT_LAYER_SIZE);
        //PrintFloatTensor("out_spk", out_spk, MY_MEM_U_OUTPUT_LAYER_SIZE);


        //printf("Just before entering elementwise_add()\n");
        //elementwise_add();
        





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