
#include <stdio.h>

//#include <math.h> //for log()
#include <stdlib.h> //rand


#include "BoardInit.h"
//#include "gpio_wrapper.h"    /* GPIO wrapper for LED control */
//#include "Driver_GPIO.h"     /* GPIO driver for LED control */
//#include "GpioSignal.hpp"




#include "ethosu_driver.h"




#include "global_map.h"
#include "include/extra_funcs.h"

#include "nn_data_structure.h"
//#include "nn_models/manual_weights/test_patterns/pattern_0.h"
#include "pm.h" //SystemCoreClock



#include "nn_models/model.h"


const int DEBUG_MODE = 0;
const int VIEW_TENSORS = 0;
const int MEASURE_MODE = 0;
const int BENCHMARK_MODEL = 0;
const int CHECK_INPUT_OUTPUT = 0;
const int GET_SPK_GRAPH = 0;


const int CACHE_ENABLED = 0;
const int LAYER_WISE_UPDATE_ENABLED = 1;


int dcache_flushed = 0;
int dcache_invalidated = 0;


//double avg_inference_time_per_sample = 0;
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




double ait_mlp_run_layer = 0;

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

double ait_SCB_CleanInvalidateDCache = 0;
double ait_disable_dcache = 0;
double ait_enable_dcache = 0;










int main() {



    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();







    //Measurement unit: 1 microsecond
    SysTick_Config(SystemCoreClock/1000000);




    printf("Running NPU model\n");

    NN_Model* mlp_model = MLP_Init();

    printf("MLP INIT successful\n");
    
    MLP_Inference_test_patterns(
        mlp_model,
        test_input_0,
        test_target_0,
        test_input_0_NUM_SAMPLES,
        1
    );



        



    // Show inference speed
    //avg_inference_time_per_sample /= (double)test_input_0_NUM_SAMPLES;
    //ait_mlp_run_layer /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_time_step_loop /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_MLP_Inference_test_pattern_while_loop /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_MLP_Inference_test_pattern /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_MLP_Run_Layer /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_run_cms /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_ethosu_invoke_v3 /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_start_inference /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_ethosu_invoke_async /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_ethosu_wait /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_post_inference_end /= (double)test_input_0_NUM_SAMPLES;
    //avg_inference_time_invalidate_ethosu_dcache /= (double)test_input_0_NUM_SAMPLES;

    //printf("avg_inference_time_per_sample:\t\t\t\t\t%f us\n", avg_inference_time_per_sample);
    //printf("avg_inference_time_time_step:\t\t\t\t\t%f us\n", avg_inference_time_time_step_loop);
    //printf("avg_inference_time_MLP_Inference_test_pattern_while_loop:\t%f us\n", avg_inference_time_MLP_Inference_test_pattern_while_loop);
    //printf("avg_inference_time_MLP_Inference_test_pattern:\t\t\t%f us\n", avg_inference_time_MLP_Inference_test_pattern);
    //printf("avg_inference_time_MLP_Run_Layer:\t\t\t\t%f us\n", avg_inference_time_MLP_Run_Layer);
    //printf("avg_inference_time_run_cms:\t\t\t\t\t%f us\n", avg_inference_time_run_cms);
    //printf("avg_inference_time_ethosu_invoke_v3:\t\t\t\t%f us\n", avg_inference_time_ethosu_invoke_v3);
    //printf("avg_inference_time_start_inference:\t\t\t\t%f us\n", avg_inference_time_start_inference);
    //printf("avg_inference_time_ethosu_invoke_async:\t\t\t\t%f us\n", avg_inference_time_ethosu_invoke_async);
    //printf("avg_inference_time_ethosu_wait:\t\t\t\t%f us\n", avg_inference_time_ethosu_wait);
    //printf("avg_inference_time_post_inference_end:\t\t\t\t%f us\n", avg_inference_time_post_inference_end);
    //printf("avg_inference_time_invalidate_ethosu_dcache:\t\t\t\t%f\n", avg_inference_time_invalidate_ethosu_dcache);


    /* This is needed to measure power
    ait_reset_model_for_new_sample /= (double)test_input_0_NUM_SAMPLES;
    ait_set_test_pattern_pointer_to_model /= (double)test_input_0_NUM_SAMPLES;
    ait_get_time_since_last_update /= (double)test_input_0_NUM_SAMPLES;
    ait_ethosu_reserve_driver /= (double)test_input_0_NUM_SAMPLES;
    ait_check_npu_nn_op_validity /= (double)test_input_0_NUM_SAMPLES;
    ait_process_cms_preamble /= (double)test_input_0_NUM_SAMPLES;
    ait_verify_base_addr /= (double)test_input_0_NUM_SAMPLES;
    ait_ethosu_flush_dcache /= (double)test_input_0_NUM_SAMPLES;
    ait_ethosu_request_power /= (double)test_input_0_NUM_SAMPLES;
    ait_ethosu_dev_run_command_stream /= (double)test_input_0_NUM_SAMPLES;
    ait_wait_npu_task_complete_irq /= (double)test_input_0_NUM_SAMPLES;
    ait_ethosu_release_power /= (double)test_input_0_NUM_SAMPLES;
    ait_ethosu_invalidate_dcache /= (double)test_input_0_NUM_SAMPLES;
    ait_ethosu_release_driver /= (double)test_input_0_NUM_SAMPLES;

    ait_SCB_CleanInvalidateDCache /= (double)test_input_0_NUM_SAMPLES;
    ait_disable_dcache /= (double)test_input_0_NUM_SAMPLES;
    ait_enable_dcache /= (double)test_input_0_NUM_SAMPLES;
    */


    //printf("%s\n", MLP_MODEL_NAME);
    //printf("\n\n\n\n\n\n\n\n\n\n\n");
    //printf("start printing inference exe time\n");

    //printf("avg_inference_exe_time_per_sample:\t\t\t\t\t%f,\n", avg_inference_time_per_sample);
    printf("ait_mlp_run_layer, %f\n", ait_mlp_run_layer);
    //printf("_________________________________\n");
    //printf("Exe time per section report:\n");
    printf("ait_reset_model_for_new_sample, %f,\n", ait_reset_model_for_new_sample);
    printf("ait_set_test_pattern_pointer_to_model, %f,\n", ait_set_test_pattern_pointer_to_model);
    printf("ait_get_time_since_last_update, %f,\n", ait_get_time_since_last_update);
    printf("ait_ethosu_reserve_driver, %f,\n", ait_ethosu_reserve_driver);
    printf("ait_check_npu_nn_op_validity, %f,\n", ait_check_npu_nn_op_validity);
    printf("ait_process_cms_preamble, %f,\n", ait_process_cms_preamble);
    printf("ait_verify_base_addr, %f,\n", ait_verify_base_addr);
    printf("ait_ethosu_flush_dcache, %f,\n", ait_ethosu_flush_dcache);
    printf("ait_ethosu_request_power, %f,\n", ait_ethosu_request_power);
    printf("ait_ethosu_dev_run_command_stream, %f,\n", ait_ethosu_dev_run_command_stream);
    printf("ait_wait_npu_task_complete_irq, %f,\n", ait_wait_npu_task_complete_irq);
    printf("ait_ethosu_release_power, %f,\n", ait_ethosu_release_power);
    printf("ait_ethosu_invalidate_dcache, %f,\n", ait_ethosu_invalidate_dcache);
    printf("ait_ethosu_release_driver, %f,\n", ait_ethosu_release_driver);
    printf("ait_arg_max, %f,\n", ait_arg_max);

    //printf("stop printing inference exe time\n");


    //printf("ait_SCB_CleanInvalidateDCache, %f,\n", ait_SCB_CleanInvalidateDCache);
    //printf("ait_disable_dcache, %f\n", ait_disable_dcache);
    //printf("ait_enable_dcache, %f\n", ait_enable_dcache);



    printf("End of main() reached entering WFE__()\n");


    // Enter WFE for 10 secs
    delay(10000000);




    printf("Start running inference forever\n"); // For measuring power
    while (1) {
        MLP_Inference_test_patterns(
            mlp_model,
        test_input_0,
        test_target_0,
        test_input_0_NUM_SAMPLES,
            0
        );
    }
     

    MLP_Free(mlp_model);

}


