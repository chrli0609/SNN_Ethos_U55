
#include <stdio.h>


#include "BoardInit.h"
//#include "gpio_wrapper.h"    /* GPIO wrapper for LED control */
//#include "Driver_GPIO.h"     /* GPIO driver for LED control */
//#include "GpioSignal.hpp"




#include "ethosu_driver.h"
//#include "ethosu_npu_init.c"

//#include "lif_model.c"





// REMOVE THESE SOOOOOONN!!!!
//#include "../python_models/conv2d_test/saved_models/cpp_tflite_vela/tflite_model_vela.tflite.cc"
//#include <inttypes.h>

//#include "lif_model.h"
//#include "nn_test.h"
//#include "nn_test.cpp"
#include "include/nn_ops.h"
#include "include/lif_model.h"
#include "include/extra_funcs.h"




const int DEBUG_MODE = 0;



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

        //printf("doing conv2d now!!!\n");
        //conv2d(8*8*16, 4*4*16);
 

        /*
        printf("doing elementwise add now!!!\n");
        //Generate input
        //uint8_t input1[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        //uint8_t input2[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
        //uint8_t output[16];

        
        uint8_t input1[256];
        uint8_t input2[256];
        for (int i = 0; i < 256; i++) {
            input1[i] = 1;
            input2[i] = 2;
        }
        uint8_t output[256];

        elementwise_add(input1, input2, output);


        printf("output:\n");
        for (int i = 0; i < 256; i++) {
            printf("%d ", output[i]);
        }
        printf("\n");
        */


        /*
        #include "include/matmul.h"
        printf("Now doing matmul!!!");
        uint8_t input[MATMUL_INPUT_TENSOR_SIZE];
        uint8_t output[MATMUL_OUTPUT_TENSOR_SIZE];
        for (int i = 0; i < MATMUL_INPUT_TENSOR_SIZE; i++) {
            input[i] = 1;
        }
        matmul(input, output);
        //matmul_vela(input, output);


        printf("Resulting output\n");
        for (int i = 0; i < MATMUL_OUTPUT_TENSOR_SIZE; i++) {
            printf(" %d,", output[i]);
        }
        printf("\n");
        */
        



        //##############################################################
        // try doing a single layer


        
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