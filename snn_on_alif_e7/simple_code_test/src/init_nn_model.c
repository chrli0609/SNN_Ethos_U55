#include "include/init_nn_model.h"


#include <stdio.h> //printf
#include <stdlib.h> //malloc



//#include "fc_lif_layer_0.h"
//#include "fc_lif_layer_1.h"

#include "cmsis_gcc.h"
#include "fc_lif_layer_0.h"
#include "fc_lif_layer_1.h"
#include "include/extra_funcs.h" //quantize_array_float_to_int8()
#include "nn_data_structure.h"
#include "pm.h"




extern int DEBUG_MODE;


// How often we update (in ms)
#define UPDATE_PERIOD 1




//#include ""

//void init_lptmr(void) {
    //// Enable LPTMR clock
    //RCC->APB1ENR |= RCC_APB1ENR_LPTIM1EN; // Enable LPTMR clock
    
    //// Configure LPTMR for 1 Hz (1-second ticks)
    //LPTMR1->CMR = 32768 - 1; // Set compare value for 1 second (based on 32.768 kHz)
    
    //// Set LPTMR for continuous mode
    //LPTMR1->CR |= LPTMR_CR_ENABLE;
    //LPTMR1->CR |= LPTMR_CR_TEN;  // Enable timer
//}



//volatile uint32_t lptmr1_start, lptmr2_start;

//void lptmr1_start_time(void) {
    //lptmr1_start = LPTMR1->CNR;  // Snapshot LPTMR1 counter
//}

//void lptmr2_start_time(void) {
    //lptmr2_start = LPTMR1->CNR;  // Snapshot LPTMR1 counter
//}

//uint32_t lptmr1_elapsed_time(void) {
    //return LPTMR1->CNR - lptmr1_start;  // Elapsed time in 32.768 kHz ticks
//}

//uint32_t lptmr2_elapsed_time(void) {
    //return LPTMR1->CNR - lptmr2_start;  // Elapsed time in 32.768 kHz ticks
//}









//#include "core_cm55.h"

//volatile uint32_t systick_flag = 0;

//void SysTick_Handler(void) {
    //systick_flag = 1;
//}

//void sleep_1ms(void) {
    //__disable_irq();        // Just to be sure during setup
    //systick_flag = 0;
    //uint32_t reload = (SystemCoreClock / 1000) - 1;
    ////uint32_t reload = 0xFFFFFF;

    //if (reload > 0xFFFFFF) { printf("clock overflowed, capped to ~42ms\n"); reload = 0xFFFFFF; } // prevent overflow

    //SysTick->LOAD = reload;
    //SysTick->VAL = 0;
    //SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk |
                    //SysTick_CTRL_TICKINT_Msk |
                    //SysTick_CTRL_ENABLE_Msk;

    //__enable_irq();         // Enable interrupts

    //while (!systick_flag) {
        //__WFI();            // Sleep until interrupt
    //}

    //SysTick->CTRL = 0;      // Disable SysTick
//}








NNLayer* FC_LIF_Layer_Init(
    size_t tensor_arena_size,
    int8_t* tensor_arena,

    size_t in_spk_relative_addr,
    size_t bias_relative_addr,
    size_t weight_relative_addr,
    size_t v_mem_relative_addr,
    size_t time_not_updated_relative_addr,
    size_t update_nxt_layer_relative_addr,
    size_t out_spk_relative_addr,

    size_t input_layer_size,
    size_t output_layer_size,
    size_t bias_tensor_size,
    size_t weight_tensor_size,


    float in_spk_scale,
    int in_spk_zero_point,

    float v_mem_scale,
    int v_mem_zero_point,
    float time_not_updated_scale,
    int time_not_updated_zero_point,

    float out_spk_scale,
    int out_spk_zero_point
) {

    NNLayer* nnlayer = NNLayer_Init(tensor_arena, tensor_arena_size, 7);
    if (nnlayer == NULL) { printf("Error when initializing NN_layer0\n"); }

    // Manually set the relative addresses
    int8_t* in_spk_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        in_spk_relative_addr);
    int8_t* bias_arena = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        bias_relative_addr);
    int8_t* weight_arena = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        weight_relative_addr);
    int8_t* v_mem_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        v_mem_relative_addr);
    int8_t* time_not_updated_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        time_not_updated_relative_addr);
    //int8_t* tmp1_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        //FC_LIF_LAYER_0_DECAY_ADDR);
    //int8_t* tmp2_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        //FC_LIF_LAYER_0_IN_CURR_ADDR);
    int8_t* update_nxt_layer_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
            update_nxt_layer_relative_addr);
    int8_t* out_spk_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
            out_spk_relative_addr);



    // Store pointers to quantized tensors for the layer in a struct
    NNLayer_Assign(nnlayer, 0, in_spk_quant, input_layer_size, in_spk_scale, in_spk_zero_point, "in_spk_quant");

    NNLayer_Assign(nnlayer, 1, bias_arena, bias_tensor_size, 1, 0, "bias_arena");
    NNLayer_Assign(nnlayer, 2, weight_arena, weight_tensor_size, 1, 0, "weight_arena");

    NNLayer_Assign(nnlayer, 3, v_mem_quant, output_layer_size, v_mem_scale, v_mem_zero_point, "v_mem_quant");
    NNLayer_Assign(nnlayer, 4, time_not_updated_quant, output_layer_size, time_not_updated_scale, time_not_updated_zero_point, "time_not_updated_quant");

        
    // Tmp1 & Tmp2, no quantization params needed
    //NNLayer_Assign(nnlayer, 7, tmp1_quant, output_layer_size, 1, 0, "tmp1_quant");
    //NNLayer_Assign(nnlayer, 8, tmp2_quant, output_layer_size, 1, 0, "tmp2_quant");
        
    // Output
    NNLayer_Assign(nnlayer, 5, update_nxt_layer_quant, 1, 1, 0, "update_nxt_layer_quant");
    NNLayer_Assign(nnlayer, 6, out_spk_quant, output_layer_size, out_spk_scale, out_spk_zero_point, "out_spk_quant");






    //3. Assign default values to V_mem
    float v_mem [output_layer_size];
    for (size_t i = 0; i < output_layer_size; i++) {
        v_mem[i] = 0;
    }
    quantize_array_float_to_int8(v_mem, nnlayer->tensor_ptrs[3], output_layer_size, v_mem_scale, v_mem_zero_point);


    // Assign default value to Time_not_updated
    float time_not_updated [output_layer_size];
    for (size_t i = 0; i < output_layer_size; i++) {
        time_not_updated[i] = 0;
    }
    quantize_array_float_to_int8(time_not_updated, nnlayer->tensor_ptrs[4], output_layer_size, time_not_updated_scale, time_not_updated_zero_point);


    // Assign layer input and output (so other layers know where to read and write from)
    nnlayer->input = in_spk_quant;
    nnlayer->output = out_spk_quant;


    return nnlayer;


}







NN_Model* MLP_Init() {


    // 1. Allocate for Total Arena Tensor on Heap
    size_t total_arena_tensor_size = FC_LIF_LAYER_0_TENSOR_ARENA_SIZE + FC_LIF_LAYER_1_TENSOR_ARENA_SIZE;
    int8_t* total_arena_tensor = (int8_t*)aligned_malloc(total_arena_tensor_size, MEM_ALIGNMENT);


    int8_t* nnlayer0_tensor_arena_start_ptr = total_arena_tensor;
    int8_t* nnlayer1_tensor_arena_start_ptr = (int8_t*)((size_t)total_arena_tensor+(FC_LIF_LAYER_0_TENSOR_ARENA_SIZE - FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE));

    // Do this for each layer we have
    // First NNLayer
    printf("about to init nnlayer0\n");
    NNLayer* nnlayer0_fc_lif = FC_LIF_Layer_Init(
        FC_LIF_LAYER_0_TENSOR_ARENA_SIZE,
        nnlayer0_tensor_arena_start_ptr,
        
        FC_LIF_LAYER_0_IN_SPK_ADDR,
        FC_LIF_LAYER_0_BIAS_ADDR,
        FC_LIF_LAYER_0_WEIGHT_ADDR,
        FC_LIF_LAYER_0_V_MEM_ADDR,
        FC_LIF_LAYER_0_TIME_NOT_UPDATED_ADDR,
        FC_LIF_LAYER_0_UPDATE_NXT_LAYER_ADDR,
        FC_LIF_LAYER_0_OUT_SPK_ADDR,
    


        FC_LIF_LAYER_0_INPUT_LAYER_SIZE,
        FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE,
        FC_LIF_LAYER_0_BIAS_LEN,
        FC_LIF_LAYER_0_WEIGHT_LEN,


        FC_LIF_LAYER_0_IN_SPK_SCALE,
        FC_LIF_LAYER_0_IN_SPK_ZERO_POINT,

        FC_LIF_LAYER_0_V_MEM_SCALE,
        FC_LIF_LAYER_0_V_MEM_ZERO_POINT,
        FC_LIF_LAYER_0_TIME_NOT_UPDATED_SCALE,
        FC_LIF_LAYER_0_TIME_NOT_UPDATED_ZERO_POINT,

        FC_LIF_LAYER_0_OUT_SPK_SCALE,
        FC_LIF_LAYER_0_OUT_SPK_ZERO_POINT

    );

    printf("About to init nnlayer1\n");
    NNLayer* nnlayer1_fc_lif = FC_LIF_Layer_Init(
        FC_LIF_LAYER_1_TENSOR_ARENA_SIZE,
        nnlayer1_tensor_arena_start_ptr,
        
        FC_LIF_LAYER_1_IN_SPK_ADDR,
        FC_LIF_LAYER_1_BIAS_ADDR,
        FC_LIF_LAYER_1_WEIGHT_ADDR,
        FC_LIF_LAYER_1_V_MEM_ADDR,
        FC_LIF_LAYER_1_TIME_NOT_UPDATED_ADDR,
        FC_LIF_LAYER_1_UPDATE_NXT_LAYER_ADDR,
        FC_LIF_LAYER_1_OUT_SPK_ADDR,
    


        FC_LIF_LAYER_1_INPUT_LAYER_SIZE,
        FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE,
        FC_LIF_LAYER_1_BIAS_LEN,
        FC_LIF_LAYER_1_WEIGHT_LEN,


        FC_LIF_LAYER_1_IN_SPK_SCALE,
        FC_LIF_LAYER_1_IN_SPK_ZERO_POINT,

        FC_LIF_LAYER_1_V_MEM_SCALE,
        FC_LIF_LAYER_1_V_MEM_ZERO_POINT,
        FC_LIF_LAYER_1_TIME_NOT_UPDATED_SCALE,
        FC_LIF_LAYER_1_TIME_NOT_UPDATED_ZERO_POINT,

        FC_LIF_LAYER_1_OUT_SPK_SCALE,
        FC_LIF_LAYER_1_OUT_SPK_ZERO_POINT

    );



    //3. Connect the models together to form a linked list
    nnlayer0_fc_lif->next_layer = nnlayer1_fc_lif;
    nnlayer1_fc_lif->next_layer = NULL;
    



    // 3. Create NN_Model
    NN_Model* mlp_model = NN_Model_Init(total_arena_tensor, nnlayer0_fc_lif);


    return mlp_model;
}


int MLP_Quantize_Inputs(NN_Model* mlp_model, float* in_spk, float* v_mem, float* time_not_updated) {

    NNLayer* nnlayer = mlp_model->first_nnlayer;

    // Quantize
    quantize_array_float_to_int8(in_spk, nnlayer->tensor_ptrs[0], FC_LIF_LAYER_0_INPUT_LAYER_SIZE, FC_LIF_LAYER_0_IN_SPK_SCALE, FC_LIF_LAYER_0_IN_SPK_ZERO_POINT);
    //quantize_array_float_to_int8(ln_beta, nnlayer->tensor_ptrs[3], FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE,FC_LIF_LAYER_0_LN_BETA_SCALE, FC_LIF_LAYER_0_LN_BETA_ZERO_POINT);
    //quantize_array_float_to_int8(vth, nnlayer->tensor_ptrs[4], FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE,FC_LIF_LAYER_0_VTH_SCALE, FC_LIF_LAYER_0_VTH_ZERO_POINT);
    quantize_array_float_to_int8(v_mem, nnlayer->tensor_ptrs[3], FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE, FC_LIF_LAYER_0_V_MEM_SCALE, FC_LIF_LAYER_0_V_MEM_ZERO_POINT);
    quantize_array_float_to_int8(time_not_updated, nnlayer->tensor_ptrs[4], FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE,FC_LIF_LAYER_0_TIME_NOT_UPDATED_SCALE, FC_LIF_LAYER_0_TIME_NOT_UPDATED_ZERO_POINT);

    return 0;
}



#include "include/nn_ops.h"
int MLP_Run_Layer(
    int8_t* tensor_arena,
    size_t tensor_arena_size,

    const uint8_t* command_stream,
    size_t command_stream_size,
    const int8_t* weight_tensor,
    size_t weight_tensor_size,

    const int8_t* lif_param,
    size_t lif_param_size,
    const int8_t* exp_lut,
    size_t exp_lut_size

)
{



  
    


    if (DEBUG_MODE) {

        // print values
        printf("BEFORE INVOKE\n");
        PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
    }



    // Assign base addrs
    const size_t num_tensors = 5;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] = (uint64_t)(intptr_t)weight_tensor;   // Model weights
    base_addrs[1] = (uint64_t)(intptr_t)tensor_arena;   // Tensor arena pointer
    base_addrs[2] = (uint64_t)(intptr_t)tensor_arena;   // Fast scratch, same as tensor arena for now
    base_addrs[3] = (uint64_t)(intptr_t)lif_param;
    base_addrs[4] = (uint64_t)(intptr_t)exp_lut;

    base_addrs_size[0] = weight_tensor_size;
    base_addrs_size[1] = tensor_arena_size;
    base_addrs_size[2] = tensor_arena_size;
    base_addrs_size[3] = lif_param_size;
    base_addrs_size[4] = exp_lut_size;


    // Sanity check to ensure num_tensors matches length of tensors
    if (sizeof(base_addrs) / sizeof(base_addrs[0]) != num_tensors) { printf("num_tensors does not match base_addrs length\n"); return -1;}
    if (sizeof(base_addrs_size) / sizeof(base_addrs_size[0]) != num_tensors) { printf("num_tensors does not match base_addrs_size length\n"); return -1;}




    // Run NPU commands
    if(run_cms(command_stream, command_stream_size, base_addrs, base_addrs_size, num_tensors) != 0) {
        printf("run_cms call failed\n");
        return -1;
    }


    if (DEBUG_MODE) {
        //print tensor values after
        printf("AFTER INVOKE\n");
        PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
    }


    return 0;



}





uint32_t volatile ms_ticks = 0;
void SysTick_Handler (void) {
    ms_ticks++;
}

void delay(uint32_t nticks){
    uint32_t c_ticks;

    c_ticks = ms_ticks;
    while((ms_ticks - c_ticks) < nticks) __WFE() ;
}



uint32_t start_timer() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    
    uint32_t start = DWT->CYCCNT;

    return start;
    
}


uint32_t debug_start_timer(void) {
    return ms_ticks;
}

uint32_t debug_end_timer(uint32_t start_tick) {
    
    //Current time - the time we started --> time elapsed
    uint32_t elapsed_ticks = ms_ticks - start_tick;
    return elapsed_ticks;

}

float end_timer(uint32_t start) {

    uint32_t end = DWT->CYCCNT;
    uint32_t elapsed_cycles = end - start;
    float elapsed_ms = (float)elapsed_cycles / (SystemCoreClock / 1000.0f);

    if (DEBUG_MODE) {
        printf("Debug tool: elapsed_ms = %f\n", elapsed_ms);
    }

    return elapsed_ms;
}


#include "include/nn_ops.h"

int MLP_Inference(
    NN_Model* mlp_model,

    float** in_spk_arr,
    size_t in_spk_arr_len,

    float* out_spk
) {


    // Set to Milimeter increase
    //try setting each tick to 
    SysTick_Config(SystemCoreClock/1000000);



    //uint32_t start, end;
    //CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    //DWT->CYCCNT = 0;
    //DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    
    //start = DWT->CYCCNT;
    ////sleep_200ms();
    //delay(200);
    //end = DWT->CYCCNT;
    //uint32_t elapsed_cycles = end - start;
    //float elapsed_ms = (float)elapsed_cycles / (SystemCoreClock / 1000.0f);

    //printf("Debug tool: elapsed_ms = %f\n", elapsed_ms);



    // Measure system
    printf("Num neurons = %d\n", FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE);


        // First Layer
        NNLayer* nnlayer0 = mlp_model->first_nnlayer;
        // Second Layer
        NNLayer* nnlayer1 = nnlayer0->next_layer;


        

        // For debugging
        uint32_t debug_timer_start; float debug_timer_elapsed_ms;

        // Timer temp variables
        uint32_t start;
        float elapsed_ms;


        // Init timer
        uint32_t start_layer0 = 0;
        uint32_t start_layer1 = 0;
        float time_not_updated_layer0_val, time_not_updated_layer1_val;


        // Start next cycle
        float* in_spk;
        size_t it = 0;

        while (it < in_spk_arr_len) {

            printf("it: %d\n", it);


            // Set up input spikes for this iteration
            in_spk = in_spk_arr[it];



            // Start measuring time
            if (DEBUG_MODE) { debug_timer_start = start_timer(); }
            start = start_timer();



            // Quantize Input in_spk
            quantize_array_float_to_int8(in_spk, nnlayer0->tensor_ptrs[0], FC_LIF_LAYER_0_INPUT_LAYER_SIZE, FC_LIF_LAYER_0_IN_SPK_SCALE, FC_LIF_LAYER_0_IN_SPK_ZERO_POINT);



         


            // Update how long it was we updated layer0 last
            // Elapsed_time since last update
            //mult by 1000 to get back from micro sec --> ms
            time_not_updated_layer0_val = 1000*end_timer(start_layer0);
            printf("time_not_updated_layer0_val: %f\n", time_not_updated_layer0_val);

            // layer0
            float time_not_updated_layer0 [FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE];
            for (size_t i = 0; i < FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE; i++) {
                time_not_updated_layer0[i] = time_not_updated_layer0_val;
            }
            quantize_array_float_to_int8(time_not_updated_layer0, nnlayer0->tensor_ptrs[4], FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE,FC_LIF_LAYER_0_TIME_NOT_UPDATED_SCALE, FC_LIF_LAYER_0_TIME_NOT_UPDATED_ZERO_POINT);

            //DEBUG: Check Tensor Arena Values Before NPU OP
            if (DEBUG_MODE) { 
                size_t in_spk_sum = 0;
                for (size_t i = 0; i < FC_LIF_LAYER_0_INPUT_LAYER_SIZE; i++) { in_spk_sum += in_spk[i]; }
                printf("In_spk_sum: %d\n", in_spk_sum);
                NNLayer_DequantizeAndPrint(nnlayer0);
            }


            uint32_t measure_layer0_start = debug_start_timer();
            // MLP Run First Layer
            MLP_Run_Layer(
                nnlayer0->tensor_arena,
                nnlayer0->tensor_arena_size,

                Getfc_lif_layer_0CMSPointer(),
                Getfc_lif_layer_0CMSLen(),
                Getfc_lif_layer_0WeightsPointer(),
                Getfc_lif_layer_0WeightsLen(),

                Getfc_lif_layer_0LIFParamPointer(),
                Getfc_lif_layer_0LIFParamLen(),
                Getfc_lif_layer_0LUTPointer(),
                Getfc_lif_layer_0LUTLen()
            );
            uint32_t measure_layer0_elapsed_ticks = debug_end_timer(measure_layer0_start);
            printf("Ticks elapsed for layer once in it: %d = %d\n", it, measure_layer0_elapsed_ticks);
            //printf("Just printed time it takes to compute 1 layer on NPU: %d\n", measure_layer0_elapsed_ticks);

            // Start timer
            start_layer0 = start_timer();
            


        
            // Check resulting Tensor Arena Values after NPU OP
            NNLayer_DequantizeAndPrint(nnlayer0);



            /*

            // Had at least 1 spike in layer0 --> run next layer
            if (((int8_t)*(nnlayer0->tensor_ptrs[5])) == 127) {

                //mydebug
                //printf("nnlayer1->input:\n");
                //if (nnlayer1->input != nnlayer1->tensor_ptrs[0]) { printf(" ITS SO JOEVER\n");} else {printf("nnlayer1->input == nnlayer1->tensor_ptrs[0]\n");}
                //if (nnlayer1->input != nnlayer0->output) { printf(" ITS SO JOEVER\n");} else {printf("nnlayer1->input == nnlayer0->output\n");}
                //if (nnlayer1->tensor_arena != nnlayer0->tensor_ptrs[6]) { printf("ITS SO JOEVER\n"); } else { printf("nnlayer1->tensor_arena == nnlayer0->tensor_ptrs[6]\n");}


                // Update how long it was we updated layer0 last
                time_not_updated_layer1_val = end_timer(start_layer1);

                float time_not_updated_layer1 [FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE];
                for (size_t i = 0; i < FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE; i++) {
                    //time_not_updated_layer1[i] = dif_layer1;
                    time_not_updated_layer1[i] = time_not_updated_layer1_val;
                }
                quantize_array_float_to_int8(time_not_updated_layer1, nnlayer1->tensor_ptrs[4], FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE,FC_LIF_LAYER_1_TIME_NOT_UPDATED_SCALE, FC_LIF_LAYER_1_TIME_NOT_UPDATED_ZERO_POINT);

                //printf("nnlayer1:\n");
                if (DEBUG_MODE) { NNLayer_DequantizeAndPrint(nnlayer1); }

                printf("starting MLP RUN Layer1 now\n");
                MLP_Run_Layer(
                    nnlayer1->tensor_arena,
                    nnlayer1->tensor_arena_size,

                    Getfc_lif_layer_1CMSPointer(),
                    Getfc_lif_layer_1CMSLen(),
                    Getfc_lif_layer_1WeightsPointer(),
                    Getfc_lif_layer_1WeightsLen(),

                    Getfc_lif_layer_1LIFParamPointer(),
                    Getfc_lif_layer_1LIFParamLen(),
                    Getfc_lif_layer_1LUTPointer(),
                    Getfc_lif_layer_1LUTLen()
                );                
                start_layer1 = start_timer();

                NNLayer_DequantizeAndPrint(nnlayer1);

            } else if (((int8_t)*(nnlayer0->tensor_ptrs[5])) == -128) {
                printf("No spike, skipping layer1 computation\n");
            } else { printf("ERRORRRRRRR!!!!!!!!!!!! UNEXPECTED VALUE FOUND IN UPDATE_NXT_LAYER\n"); }


 
            // For debug
            //int8_t* tmp1 = (int8_t*)(((size_t)nnlayer1->tensor_arena) + FC_LIF_LAYER_1_DECAYED_MEM_ADDR);
            //float tmp1_float [FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE];
            //for (size_t i = 0; i < FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE; i++) {

                //dequantize_array_int8_to_float(tmp1, tmp1_float, FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE, FC_LIF_LAYER_1_DECAYED_MEM_SCALE, FC_LIF_LAYER_1_DECAYED_MEM_ZERO_POINT);
                //printf("%f, ", tmp1_float[i]);
            //}
            //int8_t* tmp2 = (int8_t*)(((size_t)nnlayer1->tensor_arena) + FC_LIF_LAYER_1_IN_CURR_ADDR);
            //float tmp2_float [FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE];
            //for (size_t i = 0; i < FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE; i++) {

                //dequantize_array_int8_to_float(tmp2, tmp2_float, FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE, FC_LIF_LAYER_1_IN_CURR_SCALE, FC_LIF_LAYER_1_IN_CURR_ZERO_POINT);
                //printf("%d, ", tmp2[i]);
                ////printf("%f, ", tmp2_float[i]);
            //}

            */




            it++;


            // Delay before starting next layer
            elapsed_ms = end_timer(start);
            float remaining_time = UPDATE_PERIOD - elapsed_ms; 
            if (remaining_time > 0) { delay(remaining_time); }
            else { printf("Warning: computation time > update_period --> computation will lag behind\n"); }

            //debug
            if (DEBUG_MODE) { end_timer(debug_timer_start); }


        }



        

}






int MLP_Free(NN_Model* mlp_model) {

    // Deallocate total tensor arena
    aligned_free(mlp_model->total_tensor_arena);
    mlp_model->total_tensor_arena = NULL;

    // Deallocate NNLayers
    NNLayer_Free(mlp_model->first_nnlayer);
    mlp_model->first_nnlayer = NULL;

    // Deallocate NN_Model
    free(mlp_model);
    mlp_model = NULL;


    return 0;

}









