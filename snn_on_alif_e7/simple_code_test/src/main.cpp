
//#include <iostream>
#include <stdio.h>
#include <string.h> //for memset()


#include "BoardInit.hpp"

#include "gpio_wrapper.h"    /* GPIO wrapper for LED control */
#include "Driver_GPIO.h"     /* GPIO driver for LED control */
#include "GpioSignal.hpp"

#include <stdio.h>



#include "ethosu_driver.h"
//#include "ethosu_npu_init.c"

//#include "lif_model.c"
//#include "conv2d_model.hpp"
//#include "conv2d_vela.hpp"
#include "maxpool2d_vela.hpp"
//#include "maxpool2d_translated.hpp"

#include "copy_conv2d_vela.hpp"


//#include "../ethosu_compiler/output/conv2d_doc_ex_translated.hpp"
//#include "../ethosu_compiler/output/conv2d_my_translated.hpp"




// REMOVE THESE SOOOOOONN!!!!
//#include "../python_models/conv2d_test/saved_models/cpp_tflite_vela/tflite_model_vela.tflite.cc"
//#include <inttypes.h>

//#include "lif_model.h"
//#include "nn_test.h"
//#include "nn_test.cpp"
#include "nn_ops.cpp"


//#include <unistd.h> // notice this! you need it!


#define REGISTER_ADDRESS    0x20020200
#define REGISTER            (*(volatile uint8_t*)REGISTER_ADDRESS)


//void write_to_register(uint8_t value) {
//    REGISTER = value;
//}




#define _GET_DRIVER_REF(ref, peri, chan) \
    extern ARM_DRIVER_##peri Driver_##peri##chan; \
    static ARM_DRIVER_##peri * ref = &Driver_##peri##chan;
#define GET_DRIVER_REF(ref, peri, chan) _GET_DRIVER_REF(ref, peri, chan)


// Update to use the GPIO associated with P15_1 (W2)
//GET_DRIVER_REF(gpio_led, GPIO, 15);
GET_DRIVER_REF(gpio_b, GPIO, BOARD_LEDRGB0_B_GPIO_PORT);
//GET_DRIVER_REF(gpio_r, GPIO, BOARD_LEDRGB0_R_GPIO_PORT);

#define LED_PIN_NO  1









int main() {


    // Initialize the led
    gpio_b->Initialize(BOARD_LEDRGB0_B_PIN_NO, NULL);
    gpio_b->PowerControl(BOARD_LEDRGB0_B_PIN_NO, ARM_POWER_FULL);
    gpio_b->SetDirection(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
    gpio_b->SetValue(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);

    //gpio_r->Initialize(BOARD_LEDRGB0_R_PIN_NO, NULL);
    //gpio_r->PowerControl(BOARD_LEDRGB0_R_PIN_NO, ARM_POWER_FULL);
    //gpio_r->SetDirection(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
    //gpio_r->SetValue(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);


    //gpio_led->Initialize(LED_PIN_NO, NULL);
    //gpio_led->PowerControl(LED_PIN_NO, ARM_POWER_FULL);
    //gpio_led->SetDirection(LED_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);


    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

    //gpio_led->SetValue(LED_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);

    // Initialize LEDs


    //write_to_register(3);

    
    



    /*
    uint8_t input_data[256];
    uint8_t output_data[128];



    // Neuron parameters
    float w = 0.4;
    float beta = 0.819;

    //Initialize values
    float membrane_voltage = 0;
    */
    
    

    //while (1) {

        printf("============ new iteration start =======================\n");

        //const uint8_t* whole_ex_cmd_stream = GetModelPointer();
        //printf("example conv2d cms: %p\n", whole_ex_cmd_stream);
        //printf("the weights should be under: \n");
        //for (size_t i=2208; i < GetModelLen(); i++) {
        //    printf("%"PRIu8"\n", whole_ex_cmd_stream[i]);
        //}
        
        //printf("doing max pooling now!!!\n");
        //maxpool2d(8*8*16, 4*4*16);

        printf("doing conv2d now!!!\n");
        //conv2d(8*8*16, 4*4*16, 1360);
        conv2d(8*8*16, 4*4*16);
        //conv2d(8*8*16, 4*4*16, 1360+2208);
        //conv2d(8*8*16, 4*4*16, 2);

        printf("exited to while loop in main\n");




        

        //################################################################

        


        //uint8_t input_val = 256;

        //printf("the value im going to write to register is: %d\n", input_val);

        //REGISTER = input_val;


        //printf("Register value: %"PRIu32"\n", REGISTER);
        //printf("Register value: %d\n", REGISTER);

 
        
        //gpio_led->SetValue(LED_PIN_NO, GPIO_PIN_OUTPUT_STATE_TOGGLE)
        gpio_b->SetValue(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_OUTPUT_STATE_TOGGLE);

        
        /*
        float input_arr[10] = {0, 0, 1, 1, 1, 0, 1, 0.5, 1, 0.5};

        int input_arr_len = *(&input_arr + 1) - input_arr;


        struct myTuple curr_state;
        struct myTuple next_state;
        for (int i = 0; i < input_arr_len; i++) {

            next_state = leaky_integrate_fire(curr_state.membrane_voltage, input_arr[i], w, beta, 1);

            //printf("Spike: %d\t%.6f\n", curr_state.spike, curr_state.membrane_voltage);

            curr_state = next_state;

        }
        */


        printf("we are at end of while loop\n");
        

    //}

}