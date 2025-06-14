
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
//#include "nn_models/spk_mnist_784x32x10/model.h"
//#include "nn_models/nmnist_784x64x64x10/model.h"
//#include "nn_models/nmnist_784x32x32x32x10/model.h"

#include "nn_models/model.h"


const int DEBUG_MODE = 0;
const int VIEW_TENSORS = 0;
const int MEASURE_MODE = 0;
const int BENCHMARK_MODEL = 0;
const int CHECK_INPUT_OUTPUT = 0;
const int GET_SPK_GRAPH = 0;


double avg_inference_time_per_sample = 0;
double avg_inference_time_time_step_loop = 0;
double avg_inference_time_MLP_Inference_test_pattern_while_loop = 0;
double avg_inference_time_MLP_Inference_test_pattern = 0;
double avg_inference_time_MLP_Run_Layer = 0;
double avg_inference_time_run_cms = 0;
double avg_inference_time_ethosu_invoke_v3 = 0;
double avg_inference_time_start_inference = 0;
double avg_inference_time_ethosu_invoke_async = 0;
double avg_inference_time_ethosu_wait = 0;
double avg_inference_time_post_inference_end = 0;
double avg_inference_time_invalidate_ethosu_dcache = 0;




// only defined so we dont get syntax error in ethosu_driver.c
double ait_reset_model_for_new_sample = 0;
double ait_set_test_pattern_pointer_to_model = 0;
double ait_get_time_since_last_update = 0;
double ait_ethosu_reserve_driver = 0;
double ait_check_npu_nn_op_validity = 0;
double ait_process_cms_preamble = 0;
double ait_verify_base_addr = 0;
double ait_ethosu_flush_dcache = 0;
double ait_ethosu_request_power = 0;
double ait_ethosu_dev_run_command_stream = 0;
double ait_wait_npu_task_complete_irq = 0;
double ait_ethosu_release_power = 0;
double ait_ethosu_invalidate_dcache = 0;
double ait_ethosu_release_driver = 0;
double ait_arg_max = 0;

int main() {

    printf("Just reflashed!!! ______________________________________________________\n");


    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();







    //Measurement unit: 1 microsecond
    SysTick_Config(SystemCoreClock/1000000);


    NN_Model* mlp_model = MLP_Init();
    


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




     


    //int8_t out_spk [MLP_OUTPUT_LAYER_SIZE];
    int8_t out_spk [mlp_model->output_size];



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
        test_input_0_NUM_SAMPLES,

        mlp_model->num_time_steps,

        out_spk
    );


    MLP_Free(mlp_model);

        



    // Show inference speed
    avg_inference_time_per_sample /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_time_step_loop /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_MLP_Inference_test_pattern_while_loop /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_MLP_Inference_test_pattern /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_MLP_Run_Layer /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_run_cms /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_ethosu_invoke_v3 /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_start_inference /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_ethosu_invoke_async /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_ethosu_wait /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_post_inference_end /= (double)test_input_0_NUM_SAMPLES;
    avg_inference_time_invalidate_ethosu_dcache /= (double)test_input_0_NUM_SAMPLES;

    printf("avg_inference_time_per_sample:\t\t\t\t\t%f us\n", avg_inference_time_per_sample);
    printf("avg_inference_time_time_step:\t\t\t\t\t%f us\n", avg_inference_time_time_step_loop);
    printf("avg_inference_time_MLP_Inference_test_pattern_while_loop:\t%f us\n", avg_inference_time_MLP_Inference_test_pattern_while_loop);
    printf("avg_inference_time_MLP_Inference_test_pattern:\t\t\t%f us\n", avg_inference_time_MLP_Inference_test_pattern);
    printf("avg_inference_time_MLP_Run_Layer:\t\t\t\t%f us\n", avg_inference_time_MLP_Run_Layer);
    printf("avg_inference_time_run_cms:\t\t\t\t\t%f us\n", avg_inference_time_run_cms);
    printf("avg_inference_time_ethosu_invoke_v3:\t\t\t\t%f us\n", avg_inference_time_ethosu_invoke_v3);
    printf("avg_inference_time_start_inference:\t\t\t\t%f us\n", avg_inference_time_start_inference);
    printf("avg_inference_time_ethosu_invoke_async:\t\t\t\t%f us\n", avg_inference_time_ethosu_invoke_async);
    printf("avg_inference_time_ethosu_wait:\t\t\t\t%f us\n", avg_inference_time_ethosu_wait);
    printf("avg_inference_time_post_inference_end:\t\t\t\t%f us\n", avg_inference_time_post_inference_end);
    printf("avg_inference_time_invalidate_ethosu_dcache:\t\t\t\t%f\n", avg_inference_time_invalidate_ethosu_dcache);


    printf("End of main() reached, entering WFE__()\n");


    // Enter WFE for 10 secs
    delay(10000000);



}




