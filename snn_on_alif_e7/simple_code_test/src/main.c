
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

        PrintFloatTensor("v_mem", v_mem, MY_MEM_U_OUTPUT_LAYER_SIZE);
        PrintFloatTensor("out_spk", out_spk, MY_MEM_U_OUTPUT_LAYER_SIZE);











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
        int8_t input[MATMUL_INPUT_TENSOR_SIZE];
        int8_t output[MATMUL_OUTPUT_TENSOR_SIZE];
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

        
        

        /*
        printf("DOING ELEMENTWISE MUL RN!!\n");
        
        //Generate input
        //uint8_t input1[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        float input1[16] = { 2,-1,0.5,-0.3,0.02,1,1,1,1,1,1,1,1,1,1,1 };
        float input2[16] = { 0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,1,1,1,1,1 };
        float output[16];

        int input1_zero_point = -43;
        int input2_zero_point =-43;
        int output_zero_point = -44;

        float input1_scale = 0.011763370595872402;
        float input2_scale = 0.01174505427479744;
        float output_scale = 0.022980479523539543;
        


        int8_t input1_quant[16];
        int8_t input2_quant[16]; 
        int8_t output_quant[16];

        // Quantize
        quantize_array_float_to_int8(input1, input1_quant, 16, input1_scale, input1_zero_point);
        quantize_array_float_to_int8(input2, input2_quant, 16, input2_scale, input2_zero_point);


        PrintTensor("input1_qant", input1_quant, 16);
        PrintTensor("input2_qant", input2_quant, 16);
        


        dequantize_array_int8_to_float(input1_quant, input1, 16, input1_scale, input1_zero_point);

        // Print results
        printf("Dequantized\n");
        for (int i = 0; i < 16; i++)
        {
            printf(" %f,", input1[i]);
        }
        printf("\n");






        elementwise_mul(
            ELEMENTWISE_MUL_TENSOR_ARENA_SIZE,
            input1_quant,
            ELEMENTWISE_MUL_INPUT1_TENSOR_SIZE,
            input2_quant,
            ELEMENTWISE_MUL_INPUT2_TENSOR_SIZE,

            GetElementwiseMulCMSPointer(),
            GetElementwiseMulCMSLen(),
            GetElementwiseMulScalesPointer(),
            GetElementwiseMulScalesLen(),

            output_quant,
            ELEMENTWISE_MUL_OUTPUT_TENSOR_SIZE
        );
        
        // Dequantize
        dequantize_array_int8_to_float(output_quant, output, 16, output_scale, output_zero_point);

        // Print results
        printf("Dequantized\n");
        for (int i = 0; i < 16; i++)
        {
            printf(" %f,", output[i]);
        }
        printf("\n");
        */


    /*
        printf("Testing mem_update_npu\n");

        // Generate inputs

        float in_spk [MEM_UPDATE_PYTHON_INPUT_LAYER_SIZE] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        float v_mem [MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        float decay [MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE] = {0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8};


        // Declare outputs
        float v_mem_out [MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE];
        float out_spk [MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE];


        // Declare Quantized tensors
        int8_t in_spk_quant [MEM_UPDATE_PYTHON_INPUT_LAYER_SIZE];
        int8_t v_mem_quant [MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE];
        int8_t decay_quant [MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE];
        int8_t v_mem_out_quant [MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE];
        int8_t out_spk_quant [MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE];



        //In_spk:   scale: 0.007822568528354168, zero: -1
        //V_mem:    scale: 0.007832885719835758, zero: -128 
        //Decay:    scale: 0.003921556286513805, zero: -128



        // Set quantization parameters
        float in_spk_scale = 0.007824521511793137;
        int32_t in_spk_zero_point = -1;
        
        

        float v_mem_scale = 0.003920850344002247;
        int32_t v_mem_zero_point = -128;

        float decay_scale = 0.003919448237866163;
        int32_t decay_zero_point = -128;


        float v_mem_out_scale = 0.04384336993098259;
        int32_t v_mem_out_zero_point = -27;

        float out_spk_scale = 0.003921568859368563;
        int32_t out_spk_zero_point = -128;

        


        // Quantize
        quantize_array_float_to_int8(in_spk, in_spk_quant, MEM_UPDATE_PYTHON_INPUT_LAYER_SIZE, in_spk_scale, in_spk_zero_point);
        quantize_array_float_to_int8(v_mem, v_mem_quant, MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE, v_mem_scale, v_mem_zero_point);
        quantize_array_float_to_int8(decay, decay_quant, MEM_UPDATE_PYTHON_INPUT_LAYER_SIZE, decay_scale, decay_zero_point);


        printf("GetMemUpdatePythonCMSLen(): %d\n", GetMemUpdatePythonCMSLen());
        printf("GetMemUpdatePythonWeightsLen(): %d\n", GetMemUpdatePythonWeightsLen());


        // Run NPU Membrane Update
        membrane_update_npu(
            MEM_UPDATE_PYTHON_TENSOR_ARENA_SIZE,
            in_spk_quant,
            MEM_UPDATE_PYTHON_INPUT_LAYER_SIZE,
            v_mem_quant,
            MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE,
            decay_quant,
            MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE,

            GetMemUpdatePythonCMSPointer(),
            GetMemUpdatePythonCMSLen(),
            GetMemUpdatePythonWeightsPointer(),
            GetMemUpdatePythonWeightsLen(),

            v_mem_out_quant,
            MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE,
            out_spk_quant,
            MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE
        );


        // Dequant outputs
        dequantize_array_int8_to_float(v_mem_out_quant, v_mem_out, MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE, v_mem_out_scale, v_mem_out_zero_point);
        dequantize_array_int8_to_float(out_spk_quant, out_spk, MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE, out_spk_scale, out_spk_zero_point);

        // Print output
        PrintFloatTensor("v_mem_out", v_mem_out, MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE);
        PrintFloatTensor("out_spk", out_spk, MEM_UPDATE_PYTHON_OUTPUT_LAYER_SIZE);
        */





        






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