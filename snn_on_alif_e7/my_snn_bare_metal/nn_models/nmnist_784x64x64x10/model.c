#include "nn_models/nmnist_784x64x64x10/model.h"


#include <stdio.h> //printf
#include <stdlib.h> //malloc




#include "include/extra_funcs.h" //quantize_array_float_to_int8(), timer functions
#include "layers/fc_lif_layer_0.h"
#include "layers/fc_lif_layer_1.h"
//#include "model.h"
#include "nn_data_structure.h"
#include "include/nn_ops.h"     // For run_cms() in MLP_Run_Layer()
#include "nn_models/nmnist_784x64x64x10/test_patterns/pattern_0.h"

//#include "nn_models/nmnist_784x64x64x10/test_patterns/pattern_0.h"    // Import test pattern



extern int DEBUG_MODE;
extern int VIEW_TENSORS;
extern int MEASURE_MODE;
extern int BENCHMARK_MODEL;
extern int CHECK_INPUT_OUTPUT;

// How often we update (in micro sec)
#define UPDATE_PERIOD 5000





NNLayer* FC_LIF_Layer_Init(

    // Const Tensors
    const uint8_t* command_stream,
    size_t command_stream_length,

    const int8_t* bias_and_weights,
    size_t bias_and_weights_length,

    const int8_t* lif_params,
    size_t lif_params_length,

    const int8_t* luts,
    size_t luts_length,

    // Non-const tensors
    size_t tensor_arena_size,
    int8_t* tensor_arena,

    int8_t* in_spk,
    int8_t* out_spk,

    size_t bias_relative_addr,
    size_t weight_relative_addr,
    size_t v_mem_relative_addr,
    size_t time_not_updated_relative_addr,
    size_t update_nxt_layer_relative_addr,

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


    /* ____________________________________________ */
    /* 1. Initiate NNLayer Struct                   */

    NNLayer* nnlayer = NNLayer_Init(tensor_arena, tensor_arena_size, 7);
    if (nnlayer == NULL) { printf("Error when initializing NN_layer0\n"); }

    /* ___________________________________________ */




    /* ____________________________________________ */
    /* 2. Set const tensor ptrs                     */

    nnlayer->fc_lif_const_tensors.command_stream = command_stream;
    nnlayer->fc_lif_const_tensors.command_stream_length = command_stream_length;
    nnlayer->fc_lif_const_tensors.bias_and_weights = bias_and_weights;
    nnlayer->fc_lif_const_tensors.lif_params = lif_params;
    nnlayer->fc_lif_const_tensors.lif_params_length = lif_params_length;
    nnlayer->fc_lif_const_tensors.luts = luts;
    nnlayer->fc_lif_const_tensors.luts_length = luts_length;
   

    /* ___________________________________________ */



    /* ___________________________________________ */
    /* Initiate for non const tensors              */


    // Manually set the relative addresses for the non const tensors

    //int8_t* in_spk_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        //in_spk_relative_addr);
    int8_t* bias_arena = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        bias_relative_addr);
    int8_t* weight_arena = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        weight_relative_addr);
    
    // Only turn on when debugging
    //int8_t* tmp1_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        //FC_LIF_LAYER_0_DECAY_ADDR);
    //int8_t* tmp2_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        //FC_LIF_LAYER_0_IN_CURR_ADDR);
    int8_t* v_mem_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        v_mem_relative_addr);
    int8_t* time_not_updated_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
        time_not_updated_relative_addr);

    int8_t* update_nxt_layer = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
            update_nxt_layer_relative_addr);
    //int8_t* out_spk_quant = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, 
            //out_spk_relative_addr);



    // Store pointers to quantized tensors for the layer in a struct
    NNLayer_Assign(nnlayer, BIAS_TENSOR_IDX, bias_arena, bias_tensor_size, 1, 0, "bias_arena");
    NNLayer_Assign(nnlayer, WEGIHTS_TENSOR_IDX, weight_arena, weight_tensor_size, 1, 0, "weight_arena");

    // Tmp1 & Tmp2, no quantization params needed
    //NNLayer_Assign(nnlayer, 7, tmp1_quant, output_layer_size, 1, 0, "tmp1_quant");
    //NNLayer_Assign(nnlayer, 8, tmp2_quant, output_layer_size, 1, 0, "tmp2_quant");


    NNLayer_Assign(nnlayer, V_MEM_QUANT_IDX, v_mem_quant, output_layer_size, v_mem_scale, v_mem_zero_point, "v_mem_quant");
    NNLayer_Assign(nnlayer, TIME_NOT_UPDATED_QUANT_IDX, time_not_updated_quant, 1, time_not_updated_scale, time_not_updated_zero_point, "time_not_updated_quant");

        
        
    // Output
    NNLayer_Assign(nnlayer, UPDATE_NXT_LAYER_IDX, update_nxt_layer, 1, 1, 0, "update_nxt_layer");


    NNLayer_Assign(nnlayer, IN_SPK_TENSOR_IDX, in_spk, input_layer_size, in_spk_scale, in_spk_zero_point, "in_spk");
    NNLayer_Assign(nnlayer, OUT_SPK_TENSOR_IDX, out_spk, output_layer_size, out_spk_scale, out_spk_zero_point, "out_spk");





    //3. Assign default values to V_mem
    quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[V_MEM_QUANT_IDX], output_layer_size, v_mem_scale, v_mem_zero_point);

    // Assign default value to Time_not_updated
    quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[TIME_NOT_UPDATED_QUANT_IDX], 1, time_not_updated_scale, time_not_updated_zero_point);

    // Assign default value to -1
    nnlayer->time_of_previous_update = -1;




    // Assign layer input and output (so other layers know where to read and write from)
    nnlayer->input = in_spk;
    nnlayer->update_nxt = update_nxt_layer;
    nnlayer->output = out_spk;



    return nnlayer;


}







NN_Model* MLP_Init() {


    // 1. Allocate for Total Arena Tensor on Heap



    // Do this for each layer we have
    // First NNLayer
    printf("about to init nnlayer0\n");
    NNLayer* nnlayer0_fc_lif = FC_LIF_Layer_Init(

        Getfc_lif_layer_0CMSPointer(),
        Getfc_lif_layer_0CMSLen(),

        Getfc_lif_layer_0WeightsPointer(),
        Getfc_lif_layer_0WeightsLen(),

        Getfc_lif_layer_0LIFParamPointer(),
        Getfc_lif_layer_0LIFParamLen(),

        Getfc_lif_layer_0LUTPointer(),
        Getfc_lif_layer_0LUTLen(),



        FC_LIF_LAYER_0_TENSOR_ARENA_SIZE,
        nnlayer0_tensor_arena,
        
        
        nnlayer0_in_spk,
        nnlayer0_out_spk,


        FC_LIF_LAYER_0_BIAS_ADDR,
        FC_LIF_LAYER_0_WEIGHT_ADDR,
        FC_LIF_LAYER_0_V_MEM_ADDR,
        FC_LIF_LAYER_0_TIME_NOT_UPDATED_ADDR,
        FC_LIF_LAYER_0_UPDATE_NXT_LAYER_ADDR,
        
    


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
        Getfc_lif_layer_1CMSPointer(),
        Getfc_lif_layer_1CMSLen(),

        Getfc_lif_layer_1WeightsPointer(),
        Getfc_lif_layer_1WeightsLen(),

        Getfc_lif_layer_1LIFParamPointer(),
        Getfc_lif_layer_1LIFParamLen(),

        Getfc_lif_layer_1LUTPointer(),
        Getfc_lif_layer_1LUTLen(),
        FC_LIF_LAYER_1_TENSOR_ARENA_SIZE,
        nnlayer1_tensor_arena,
        
        nnlayer0_out_spk,
        nnlayer1_out_spk,

        FC_LIF_LAYER_1_BIAS_ADDR,
        FC_LIF_LAYER_1_WEIGHT_ADDR,
        FC_LIF_LAYER_1_V_MEM_ADDR,
        FC_LIF_LAYER_1_TIME_NOT_UPDATED_ADDR,
        FC_LIF_LAYER_1_UPDATE_NXT_LAYER_ADDR,
    


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

    printf("About to init nnlayer2\n");
    NNLayer* nnlayer2_fc_lif = FC_LIF_Layer_Init(
        Getfc_lif_layer_2CMSPointer(),
        Getfc_lif_layer_2CMSLen(),

        Getfc_lif_layer_2WeightsPointer(),
        Getfc_lif_layer_2WeightsLen(),

        Getfc_lif_layer_2LIFParamPointer(),
        Getfc_lif_layer_2LIFParamLen(),

        Getfc_lif_layer_2LUTPointer(),
        Getfc_lif_layer_2LUTLen(),
        FC_LIF_LAYER_2_TENSOR_ARENA_SIZE,
        nnlayer2_tensor_arena,
        
        nnlayer1_out_spk,
        nnlayer2_out_spk,

        FC_LIF_LAYER_2_BIAS_ADDR,
        FC_LIF_LAYER_2_WEIGHT_ADDR,
        FC_LIF_LAYER_2_V_MEM_ADDR,
        FC_LIF_LAYER_2_TIME_NOT_UPDATED_ADDR,
        FC_LIF_LAYER_2_UPDATE_NXT_LAYER_ADDR,
    


        FC_LIF_LAYER_2_INPUT_LAYER_SIZE,
        FC_LIF_LAYER_2_OUTPUT_LAYER_SIZE,
        FC_LIF_LAYER_2_BIAS_LEN,
        FC_LIF_LAYER_2_WEIGHT_LEN,


        FC_LIF_LAYER_2_IN_SPK_SCALE,
        FC_LIF_LAYER_2_IN_SPK_ZERO_POINT,

        FC_LIF_LAYER_2_V_MEM_SCALE,
        FC_LIF_LAYER_2_V_MEM_ZERO_POINT,
        FC_LIF_LAYER_2_TIME_NOT_UPDATED_SCALE,
        FC_LIF_LAYER_2_TIME_NOT_UPDATED_ZERO_POINT,

        FC_LIF_LAYER_2_OUT_SPK_SCALE,
        FC_LIF_LAYER_2_OUT_SPK_ZERO_POINT

    );



    //3. Connect the models together to form a linked list
    nnlayer0_fc_lif->next_layer = nnlayer1_fc_lif;
    nnlayer1_fc_lif->next_layer = nnlayer2_fc_lif;
    nnlayer2_fc_lif->next_layer = NULL;


    nnlayer0_fc_lif->update_curr = NULL;
    nnlayer1_fc_lif->update_curr = nnlayer0_fc_lif->update_nxt;
    nnlayer2_fc_lif->update_curr = nnlayer1_fc_lif->update_nxt;
    



    // 3. Create NN_Model
    NN_Model* mlp_model = NN_Model_Init(NULL, nnlayer0_fc_lif);

    return mlp_model;
}




// Not in use anymore!!!
int MLP_Quantize_Inputs(NN_Model* mlp_model, float* in_spk, float* v_mem, float* time_not_updated) {

    NNLayer* nnlayer = mlp_model->first_nnlayer;

    // Quantize
    //quantize_array_float_to_int8(in_spk, nnlayer->tensor_ptrs[0], FC_LIF_LAYER_0_INPUT_LAYER_SIZE, FC_LIF_LAYER_0_IN_SPK_SCALE, FC_LIF_LAYER_0_IN_SPK_ZERO_POINT);
    //quantize_array_float_to_int8(ln_beta, nnlayer->tensor_ptrs[3], FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE,FC_LIF_LAYER_0_LN_BETA_SCALE, FC_LIF_LAYER_0_LN_BETA_ZERO_POINT);
    //quantize_array_float_to_int8(vth, nnlayer->tensor_ptrs[4], FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE,FC_LIF_LAYER_0_VTH_SCALE, FC_LIF_LAYER_0_VTH_ZERO_POINT);
    //quantize_array_float_to_int8(v_mem, nnlayer->tensor_ptrs[3], FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE, FC_LIF_LAYER_0_V_MEM_SCALE, FC_LIF_LAYER_0_V_MEM_ZERO_POINT);
    //quantize_array_float_to_int8(time_not_updated, nnlayer->tensor_ptrs[4], FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE,FC_LIF_LAYER_0_TIME_NOT_UPDATED_SCALE, FC_LIF_LAYER_0_TIME_NOT_UPDATED_ZERO_POINT);

    return 0;
}



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
    size_t exp_lut_size,

    int8_t* input,
    size_t input_size,
    int8_t* output,
    size_t output_size
)
{



  
    


    if (DEBUG_MODE) {

        // print values
        printf("BEFORE INVOKE\n");
        //PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
    }



    // Assign base addrs
    const size_t num_tensors = 7;
    uint64_t base_addrs[num_tensors];
    size_t base_addrs_size[num_tensors];

    base_addrs[0] = (uint64_t)(intptr_t)weight_tensor;   // Model weights
    base_addrs[1] = (uint64_t)(intptr_t)tensor_arena;   // Tensor arena pointer
    base_addrs[2] = (uint64_t)(intptr_t)tensor_arena;   // Fast scratch, same as tensor arena for now
    base_addrs[3] = (uint64_t)(intptr_t)lif_param;
    base_addrs[4] = (uint64_t)(intptr_t)exp_lut;
    base_addrs[5] = (uint64_t)(intptr_t)input;
    base_addrs[6] = (uint64_t)(intptr_t)output;

    base_addrs_size[0] = weight_tensor_size;
    base_addrs_size[1] = tensor_arena_size;
    base_addrs_size[2] = tensor_arena_size;
    base_addrs_size[3] = lif_param_size;
    base_addrs_size[4] = exp_lut_size;
    base_addrs_size[5] = input_size;
    base_addrs_size[6] = output_size;


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
        //PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
    }


    return 0;



}




void reset_membrane_potential(NN_Model* mlp_model) {
    NNLayer* nnlayer = mlp_model->first_nnlayer;

    while(nnlayer != NULL) {
        quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[V_MEM_QUANT_IDX], nnlayer->tensor_sizes[V_MEM_QUANT_IDX], nnlayer->quant_params[V_MEM_QUANT_IDX].scale, nnlayer->quant_params[V_MEM_QUANT_IDX].zero_point);
        nnlayer = nnlayer->next_layer;
    }
}


void reset_out_spk(NN_Model* mlp_model) {
    NNLayer* nnlayer = mlp_model->first_nnlayer;

    while(nnlayer != NULL) {
        quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[OUT_SPK_TENSOR_IDX], nnlayer->tensor_sizes[OUT_SPK_TENSOR_IDX], nnlayer->quant_params[OUT_SPK_TENSOR_IDX].scale, nnlayer->quant_params[OUT_SPK_TENSOR_IDX].zero_point);
        nnlayer = nnlayer->next_layer;
    }
}

void reset_layer_update_time_stamp(NN_Model* mlp_model) {
    NNLayer* nnlayer = mlp_model->first_nnlayer;
    while(nnlayer != NULL) {
        nnlayer->time_of_previous_update = -1;
        nnlayer = nnlayer->next_layer;
    }

}

// Resets the model between samples, i.e. membrane potential, out_spk, and time_stamp of previous update for each layer
// Is equivalent to running reset_membrane_potential(), reset_out_spk() and reset_layer_update_time_stamp() in succession
void reset_model_for_new_sample(NN_Model* mlp_model) {
    NNLayer* nnlayer = mlp_model->first_nnlayer;
    while(nnlayer != NULL) {

        // Reset Membrane Potential
        quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[V_MEM_QUANT_IDX], nnlayer->tensor_sizes[V_MEM_QUANT_IDX], nnlayer->quant_params[V_MEM_QUANT_IDX].scale, nnlayer->quant_params[V_MEM_QUANT_IDX].zero_point);

        // Reset Out_spk
        quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[OUT_SPK_TENSOR_IDX], nnlayer->tensor_sizes[OUT_SPK_TENSOR_IDX], nnlayer->quant_params[OUT_SPK_TENSOR_IDX].scale, nnlayer->quant_params[OUT_SPK_TENSOR_IDX].zero_point);

        // Reset time stamp for previous update
        nnlayer->time_of_previous_update = -1;

        // Move to next layer
        nnlayer = nnlayer->next_layer;
    }

}






void print_dequant_int8(int8_t* arr, size_t arr_len, const char* arr_name, float scale, int zero_point) {

    printf("%s:\n", arr_name);
    for (size_t i = 0; i < arr_len; i++) {
        float dequantized_value = (arr[i] - zero_point) * scale;
        //delay(1000);
        printf("%.5f, ", dequantized_value);
    }
    printf("\n");
}





// For debugging
int global_it;

int MLP_Inference_test_patterns(
    NN_Model* mlp_model,

    volatile int8_t test_patterns[test_input_0_NUM_SAMPLES][MLP_NUM_TIME_STEPS][MLP_INPUT_LAYER_SIZE],
    volatile int8_t test_targets[test_input_0_NUM_SAMPLES],

    size_t num_samples,

    size_t num_time_steps,

    int8_t* out_spk
) {

    // Measure system
    if (MEASURE_MODE) {
        printf("Num input neurons = %d\n", FC_LIF_LAYER_0_INPUT_LAYER_SIZE);    // Sweep over layer input_size
        printf("Num output neurons = %d\n", FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE);  // Sweep over layer output_size
        //printf("Block Configuration = (%d, %d, %d)\n", FC_LIF_LAYER_0_BLOCK_CONFIG_HEIGHT, FC_LIF_LAYER_0_BLOCK_CONFIG_WIDTH, FC_LIF_LAYER_0_BLOCK_CONFIG_DEPTH);
        //printf("Block Configuration for it: 0 = %d\n", FC_LIF_LAYER_0_BLOCK_CONFIG_DEPTH);
    } 
    //if (MEASURE_MODE) { printf("Num neurons = %d\n", FC_LIF_LAYER_0_INPUT_LAYER_SIZE); }    // Sweep over input_size


    // For debugging
    size_t number_of_no_spk = 0;
    uint32_t debug_timer_start; float debug_timer_elapsed_ms;




    // For Benchmarking accuracy
    size_t correct = 0;
    size_t prediction_arr [test_input_0_NUM_SAMPLES] = { 0 };

    // For benchmarking inference speed
    double average_inference_time = 0;





    float time_steps_layer_not_updated;


    // For every input sample (in real system would be while(true) loop)

    size_t it = 0;
    while (it < num_samples) {

        if (CHECK_INPUT_OUTPUT) {
            printf("==========================\new sample!!!==========================\n");
            printf("it: %d\n", it);
        }

        global_it = it;
        


        /*___________________________________________________________
          Reset the parameters that need to be reset for every sample  */
        
        //// Reset Membrane potential between input samples
        //// Reset output spike in case last layer is never computed (because no output spikes at all from the layer before that)
        //// Reset time stamp for previous layer update
        reset_model_for_new_sample(mlp_model);


        // For storing sum of output spikes across the time steps
        size_t out_neuron_sum[MLP_OUTPUT_LAYER_SIZE] = { 0 };

        /* ________________________________________________________ */


        // Start measuring time
        if (DEBUG_MODE) { debug_timer_start = start_timer(); }
        //start = start_timer();



        // Feed the same input to the network for num_time_steps
        for (size_t time_step = 0; time_step < num_time_steps; time_step++){
            if (CHECK_INPUT_OUTPUT) {
                printf("-----------------------new time step!!!--------------------------\n");
                printf("time step: %d\n", time_step);
            }


            // Set First Layer as current layer
            NNLayer* nnlayer = mlp_model->first_nnlayer;


            // Set new input
            nnlayer->input = test_patterns[it][time_step];


            while (nnlayer != NULL) {
                

                // Had at least 1 spike in layer0 --> run next layer
                if ( nnlayer == mlp_model->first_nnlayer || ((int8_t)*(nnlayer->update_curr) == 127) ){


                    // Update how long it was we updated layer0 last
                    time_steps_layer_not_updated = time_step - nnlayer->time_of_previous_update;
                    quantize_float_scalar_to_int8_array(time_steps_layer_not_updated, nnlayer->tensor_ptrs[TIME_NOT_UPDATED_QUANT_IDX], nnlayer->tensor_sizes[TIME_NOT_UPDATED_QUANT_IDX], nnlayer->quant_params[TIME_NOT_UPDATED_QUANT_IDX].scale, nnlayer->quant_params[TIME_NOT_UPDATED_QUANT_IDX].zero_point);

                    /* DEBUG PRINTOUTS */
                    //printf("nnlayer:\n");
                    if (VIEW_TENSORS) { printf("Pre nnlayer\n"); NNLayer_DequantizeAndPrint(nnlayer); }
                    if (CHECK_INPUT_OUTPUT) { print_dequant_int8(nnlayer->input, FC_LIF_LAYER_1_INPUT_LAYER_SIZE, "Layer1->input", FC_LIF_LAYER_1_IN_SPK_SCALE, FC_LIF_LAYER_1_IN_SPK_ZERO_POINT); }
                    if (DEBUG_MODE) { printf("starting MLP RUN Layer1 now\n"); }



                    // MLP Run Second Layer
                    MLP_Run_Layer(
                        nnlayer->tensor_arena,
                        nnlayer->tensor_arena_size,

                        nnlayer->fc_lif_const_tensors.command_stream,
                        nnlayer->fc_lif_const_tensors.command_stream_length,
                        nnlayer->fc_lif_const_tensors.bias_and_weights,
                        nnlayer->fc_lif_const_tensors.bias_and_weights_length,

                        nnlayer->fc_lif_const_tensors.lif_params,
                        nnlayer->fc_lif_const_tensors.lif_params_length,
                        nnlayer->fc_lif_const_tensors.luts,
                        nnlayer->fc_lif_const_tensors.luts_length,

                        nnlayer->input,
                        nnlayer->tensor_sizes[IN_SPK_TENSOR_IDX],
                        nnlayer->output,
                        nnlayer->tensor_sizes[OUT_SPK_TENSOR_IDX]
                    );
        

                    //start_layer1 = start_timer();
                    nnlayer->time_of_previous_update = time_step;
        
                    if (VIEW_TENSORS) { printf("Post nnlayer:\n"); NNLayer_DequantizeAndPrint(nnlayer); }
                    if (CHECK_INPUT_OUTPUT) {
                        print_dequant_int8(nnlayer->tensor_ptrs[V_MEM_QUANT_IDX], FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE, "Layer1->v_mem", FC_LIF_LAYER_1_V_MEM_SCALE, FC_LIF_LAYER_1_V_MEM_ZERO_POINT);
                        print_dequant_int8(nnlayer->output, FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE, "Layer1->output", FC_LIF_LAYER_1_OUT_SPK_SCALE, FC_LIF_LAYER_1_OUT_SPK_ZERO_POINT);
                    }


                } else if ((int8_t)*(nnlayer->update_curr) == -128) {
                    //printf("No spike, skipping layer1 computation\n");
                    //else no more to compute for this inference cycle
                    break;
                } else { printf("ERROR: Unexpected update_nxt_layer value found. Expected 127 or -128 but received: %d\n", (int8_t)*(nnlayer->update_curr)); }

                nnlayer = nnlayer->next_layer;

            }

            
            // Record the spikes (rate encoding --> use total number of spikes to determine prediction)
            //printf("output:\n");
            for (size_t i = 0; i < MLP_OUTPUT_LAYER_SIZE; i++) {
                out_neuron_sum[i] += mlp_model->output[i];

                //printf("%d: %d\n", i, mlp_model->output[i]);
            }
        
        }


        // Get the max value
        size_t max_value = 0;
        size_t max_spk_idx = 0;
        size_t neuron_sum = 0;
        for (size_t i = 0; i < MLP_OUTPUT_LAYER_SIZE; i++) {
            neuron_sum = out_neuron_sum[i]; 
            if (neuron_sum > max_value) {
                max_value = neuron_sum;
                max_spk_idx = i;
            }
        }
        printf("Prediction: %d\n", max_spk_idx);

        // Check if correct or not and add to counter
        if (max_spk_idx == (size_t)test_targets[it]) { correct += 1; }
        prediction_arr[it] = max_spk_idx;

        // Debug: Check how often we have 0 output spikes
        //if (DEBUG_MODE) {
            bool have_at_least_one_spk = false;
            for (size_t i = 0; i < MLP_OUTPUT_LAYER_SIZE; i++){
                if (out_neuron_sum[i] != 0) { have_at_least_one_spk = true; }
            }
            if (!have_at_least_one_spk) { number_of_no_spk += 1; printf("incrementing number_of_no_spk!\n");}
        //}
        


        it++;


        //// Delay before starting next layer
        //elapsed_ms = end_timer(start);
        //float remaining_time = UPDATE_PERIOD - elapsed_ms; 
        //if (remaining_time > 0) { delay(remaining_time); 
            ////printf("Slept for %f\n", remaining_time);
        //}
        //else { printf("Warning: computation time > update_period --> computation will lag behind\n"); }

        ////debug
        //if (DEBUG_MODE) { end_timer(debug_timer_start); }


    }


    printf("All Predictions:\n");
    for (size_t i = 0; i < num_samples; i++) {
        printf("%d, ", prediction_arr[i]);
    }
    printf("\n");
    // Show stats at the end
    double accuracy = (double)correct / (double)num_samples;
    printf("The total accuracy over %d input patterns is: %f\n", num_samples, accuracy);
    printf("Num samples with zero output spikes across all time steps: %d\n", number_of_no_spk);
        

}

int MLP_Inference(
    NN_Model* mlp_model,

    int8_t** in_spk_arr,
    size_t in_spk_arr_len,

    int8_t* out_spk
) {

    size_t num_time_steps = 2;





    // Measure system
    if (MEASURE_MODE) {
        printf("Num input neurons = %d\n", FC_LIF_LAYER_0_INPUT_LAYER_SIZE);    // Sweep over layer input_size
        printf("Num output neurons = %d\n", FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE);  // Sweep over layer output_size
        //printf("Block Configuration = (%d, %d, %d)\n", FC_LIF_LAYER_0_BLOCK_CONFIG_HEIGHT, FC_LIF_LAYER_0_BLOCK_CONFIG_WIDTH, FC_LIF_LAYER_0_BLOCK_CONFIG_DEPTH);
        //printf("Block Configuration for it: 0 = %d\n", FC_LIF_LAYER_0_BLOCK_CONFIG_DEPTH);
    } 
    //if (MEASURE_MODE) { printf("Num neurons = %d\n", FC_LIF_LAYER_0_INPUT_LAYER_SIZE); }    // Sweep over input_size


    // Set First Layer as current layer
    NNLayer* nnlayer0 = mlp_model->first_nnlayer;
    // Second Layer
    NNLayer* nnlayer1 = nnlayer0->next_layer;


        

    // For debugging
    uint32_t debug_timer_start; float debug_timer_elapsed_ms;

    // Timer temp variables
    uint32_t start;
    float elapsed_ms;


    float ms_time_not_updated_layer0_val, ms_time_not_updated_layer1_val;


    // Start next cycle
    int8_t* in_spk;
    size_t it = 0;


    // For every input sample (in real system would be while(true) loop)
    while (it < in_spk_arr_len) {

        //printf("it: %d\n", it);

        global_it = it;
        // Set up input spikes for this iteration
        in_spk = in_spk_arr[it];
        // For testing set the same always
        //in_spk = in_spk_arr[0];


        /*
        Reset the parameters that need to be reset for every sample
        */

        // For storing sum of output spikes across the time steps
        size_t out_neuron_sum[MLP_OUTPUT_LAYER_SIZE] = { 0 };
        // Init timer
        int32_t start_layer0 = -1;
        int32_t start_layer1 = -1;


        // Start measuring time
        if (DEBUG_MODE) { debug_timer_start = start_timer(); }
        start = start_timer();


        // Write Input in_spk
        for (size_t i = 0; i < MLP_INPUT_LAYER_SIZE; i++){
            //nnlayer0->tensor_ptrs[IN_SPK_TENSOR_IDX][i] = in_spk[i];
            nnlayer0->input[i] = in_spk[i];
        }
            
            
        // Feed the same input to the network for num_time_steps
        for (size_t time_step = 0; time_step < num_time_steps; time_step++){

        
            // Update how long it was we updated layer0 last
            // Elapsed_time since last update
            //mult by 1000 to get back from micro sec --> ms
            //ms_time_not_updated_layer0_val = 1000*end_timer(start_layer0);
            // Set this as iteration first to make sure it works



            ms_time_not_updated_layer0_val = time_step - start_layer0;
            printf("ms_time_not_updated_layer0_val: %f\n", ms_time_not_updated_layer0_val);
            // layer0 time
            float ms_time_not_updated_layer0[1] = { ms_time_not_updated_layer0_val };
            quantize_array_float_to_int8(ms_time_not_updated_layer0, nnlayer0->tensor_ptrs[TIME_NOT_UPDATED_QUANT_IDX], 1,FC_LIF_LAYER_0_TIME_NOT_UPDATED_SCALE, FC_LIF_LAYER_0_TIME_NOT_UPDATED_ZERO_POINT);



            printf("about to run layer\n");
            //DEBUG: Check Tensor Arena Values Before NPU OP
            if (VIEW_TENSORS) { 
                printf("Pre NNLayer0\n");
                size_t in_spk_sum = 0;
                for (size_t i = 0; i < FC_LIF_LAYER_0_INPUT_LAYER_SIZE; i++) { in_spk_sum += in_spk[i]; }
                printf("In_spk_sum: %d\n", in_spk_sum);
                NNLayer_DequantizeAndPrint(nnlayer0);
            }

            //uint32_t measure_layer0_start = debug_start_timer();
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
                Getfc_lif_layer_0LUTLen(),

                nnlayer0->input,
                FC_LIF_LAYER_0_INPUT_LAYER_SIZE,
                nnlayer0->output,
                FC_LIF_LAYER_0_OUTPUT_LAYER_SIZE
            );
            //uint32_t measure_layer0_elapsed_ticks = debug_end_timer(measure_layer0_start);
            //if (MEASURE_MODE) { printf("Ticks elapsed for layer once in it: %d = %d\n", it, measure_layer0_elapsed_ticks); }
            //printf("Just printed time it takes to compute 1 layer on NPU: %d\n", measure_layer0_elapsed_ticks);

            // Start timer
            //start_layer0 = start_timer();
            start_layer0 = time_step;
            


        
            // Check resulting Tensor Arena Values after NPU OP
            if (VIEW_TENSORS) {printf("Post NNLayer0\n"); NNLayer_DequantizeAndPrint(nnlayer0); }



        

            // Had at least 1 spike in layer0 --> run next layer
            if (((int8_t)*(nnlayer0->tensor_ptrs[UPDATE_NXT_LAYER_IDX])) == 127) {

                //mydebug
                //printf("nnlayer1->input:\n");
                //if (nnlayer1->input != nnlayer1->tensor_ptrs[0]) { printf(" ITS SO JOEVER\n");} else {printf("nnlayer1->input == nnlayer1->tensor_ptrs[0]\n");}
                //if (nnlayer1->input != nnlayer0->output) { printf(" ITS SO JOEVER\n");} else {printf("nnlayer1->input == nnlayer0->output\n");}
                //if (nnlayer1->tensor_arena != nnlayer0->tensor_ptrs[6]) { printf("ITS SO JOEVER\n"); } else { printf("nnlayer1->tensor_arena == nnlayer0->tensor_ptrs[6]\n");}


                // Update how long it was we updated layer0 last
                //ms_time_not_updated_layer1_val = end_timer(start_layer1);
                ms_time_not_updated_layer1_val = time_step - start_layer1;

                float ms_time_not_updated_layer1[1] = { ms_time_not_updated_layer1_val };
                quantize_array_float_to_int8(ms_time_not_updated_layer1, nnlayer1->tensor_ptrs[TIME_NOT_UPDATED_QUANT_IDX], 1,FC_LIF_LAYER_1_TIME_NOT_UPDATED_SCALE, FC_LIF_LAYER_1_TIME_NOT_UPDATED_ZERO_POINT);

                //printf("nnlayer1:\n");
                if (VIEW_TENSORS) { printf("Pre NNLayer1\n"); NNLayer_DequantizeAndPrint(nnlayer1); }

                printf("starting MLP RUN Layer1 now\n");
                // MLP Run First Layer
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
                    Getfc_lif_layer_1LUTLen(),

                    nnlayer1->input,
                    FC_LIF_LAYER_1_INPUT_LAYER_SIZE,
                    nnlayer1->output,
                    FC_LIF_LAYER_1_OUTPUT_LAYER_SIZE
                );
        
                //start_layer1 = start_timer();
                start_layer1 = time_step;
        
                if (VIEW_TENSORS) { printf("Post NNLayer1:\n"); NNLayer_DequantizeAndPrint(nnlayer1); }


            } else if (((int8_t)*(nnlayer0->tensor_ptrs[UPDATE_NXT_LAYER_IDX])) == -128) {
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

        

            /*
            Set up for reading output, show the sum of the different neuron outputs
            */

            for (size_t i = 0; i < MLP_OUTPUT_LAYER_SIZE; i++) {
                out_neuron_sum[i] += nnlayer1->output[i];
            }
            //printf("nnlayer1->output:\n");
            //for (size_t i = 0; i < MLP_OUTPUT_LAYER_SIZE; i++) {
                //printf("\t%d: %d\n", i, nnlayer1->output[i]);
            //}
        
        }


        // Now print the total sum for each neuron output
        printf("out_neuron_sum:\n");
        for (size_t i = 0; i < MLP_OUTPUT_LAYER_SIZE; i++) {
            printf("\t%d: %d\n", i, out_neuron_sum[i]);
        }



        it++;


        // Delay before starting next layer
        elapsed_ms = end_timer(start);
        float remaining_time = UPDATE_PERIOD - elapsed_ms; 
        if (remaining_time > 0) { delay(remaining_time); 
            //printf("Slept for %f\n", remaining_time);
        }
        else { printf("Warning: computation time > update_period --> computation will lag behind\n"); }

        //debug
        if (DEBUG_MODE) { end_timer(debug_timer_start); }


    }

        

}






int MLP_Free(NN_Model* mlp_model) {


    // Deallocate NNLayers
    NNLayer_Free(mlp_model->first_nnlayer);
    mlp_model->first_nnlayer = NULL;

    // Deallocate NN_Model
    free(mlp_model);
    mlp_model = NULL;


    return 0;

}









