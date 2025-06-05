
#include <stdio.h>

//#include <math.h> //for log()
#include <stdlib.h> //rand


#include "BoardInit.h"
//#include "gpio_wrapper.h"    /* GPIO wrapper for LED control */
//#include "Driver_GPIO.h"     /* GPIO driver for LED control */
//#include "GpioSignal.hpp"




#include "ethosu_driver.h"




#include "include/extra_funcs.h"

//#include "cmsis_gcc.h"
//#include "nn_models/spk_mnist_mlp/test_patterns/pattern_0.h"
#include "pm.h" //SystemCoreClock


// Include NN Model Here (Only one should be used at a time, dont forget to also change the compiled model.c file)
//#include "nn_models/single_tensor_dtcm_mlp/model.h"
//#include "nn_models/multi_tensor_sram_mlp/model.h"
//#include "nn_models/spk_mnist_mlp/model.h"
#include "nn_models/spk_mnist_784x32x10/model.h"



const int DEBUG_MODE = 0;
const int VIEW_TENSORS = 0;
const int MEASURE_MODE = 0;
const int BENCHMARK_MODEL = 0;
const int CHECK_INPUT_OUTPUT = 0;



//#include <unistd.h> // notice this! you need it!


//#define REGISTER_ADDRESS    0x20020200
//#define REGISTER            (*(volatile uint8_t*)REGISTER_ADDRESS)


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



//#define TEST_REGISTER_ADDRESS    0x02000000
//#define TEST_REGISTER            (*(volatile int8_t*)TEST_REGISTER_ADDRESS)


//void write_to_register(uint8_t value) {
//    REGISTER = value;
//}




int main() {

    printf("Just reflashed!!! ______________________________________________________\n");


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


    //static volatile __attribute__((section(".data_sram0"))) int8_t large_buffer_sram0[256];  // 256 KB in SRAM0
    //static volatile __attribute__((section("model_params_sram0"))) int8_t large_buffer_sram0[256];  // 256 KB in SRAM0
    //static volatile __attribute__((section("model_params_sram1"))) int8_t large_buffer_sram1[256];  // 256 KB in SRAM0
    //static volatile __attribute__((section("model_params_dtcm"))) int8_t large_buffer_dtcm[256];  // 256 KB in SRAM0

    //printf("Testing heap allocation\n");
    //printf("Allocated addresses for sram0 test: 0x%08X\n", large_buffer_sram0);
    //printf("Allocated addresses for dtcm test: 0x%08X\n", large_buffer_dtcm);
    //printf("Allocated addresses for sram1 test: 0x%08X\n", large_buffer_sram1);

    //printf("TEST_REGISTER_ADDRESS:  0x%08X\n", TEST_REGISTER_ADDRESS);
    //TEST_REGISTER = 2;
    //printf("TEST_REGISTER SHOULD BE 2: %d\n", TEST_REGISTER);
    //TEST_REGISTER = 44;
    //printf("TEST_REGISTER SHOULD BE 44: %d\n", TEST_REGISTER);




    //Measurement unit: 1 microsecond
    SysTick_Config(SystemCoreClock/1000000);


    NN_Model* mlp_model = MLP_Init();
    printf("Model\n");
    printf("\tFC_LIF_0: %d x %d\n", FC_LIF_LAYER_0_INPUT_LAYER_SIZE, FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE);
    printf("\tFC_LIF_1: %d x %d\n", FC_LIF_LAYER_1_INPUT_LAYER_SIZE, FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE);
    


    //srand(0);
    //size_t NUM_SAMPLES;
    //if (MEASURE_MODE) { NUM_SAMPLES = 1; }
    //else { NUM_SAMPLES = 3; }
        
    //int8_t* in_spk_arr [NUM_SAMPLES];
    //for (size_t i = 0; i < NUM_SAMPLES; i++) {

        //int8_t in_spk [MLP_INPUT_LAYER_SIZE];
        //for (size_t j = 0; j < MLP_INPUT_LAYER_SIZE; j++){
            //in_spk[j] = j % 2;
            ////if (j % 2 == 0) { in_spk[j] = -128; }
            ////else { in_spk[j] = 127; }
        //}

        //in_spk_arr[i] = in_spk;
    //}




     


    int8_t out_spk [MLP_OUTPUT_LAYER_SIZE];



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





    //MLP_Inference(
        //mlp_model,
        //in_spk_arr,
        //NUM_SAMPLES,

        //out_spk
    //);
    MLP_Inference_test_patterns(
        mlp_model,
        test_input_0,
        test_target_0,
        //test_input_0_NUM_SAMPLES,
        test_input_0_NUM_SAMPLES,

        MLP_NUM_TIME_STEPS,

        out_spk
    );


    MLP_Free(mlp_model);

        




    printf("exited to while loop in main\n");
        


        

    //################################################################

   
        
    //gpio_led->SetValue(LED_PIN_NO, GPIO_PIN_OUTPUT_STATE_TOGGLE)
    //gpio_b->SetValue(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_OUTPUT_STATE_TOGGLE);
        

}