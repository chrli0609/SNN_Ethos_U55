#include "model.h"


#include <stdio.h> //printf
#include <stdlib.h> //malloc


#include "include/nn_ops.h"     // For run_cms() in MLP_Run_Layer()
#include "include/extra_funcs.h" //quantize_array_float_to_int8(), timer functions



#include "connectivity.h"
//#include "manual_weights/test_patterns/pattern_0.h"
#include "nn_data_structure.h"
//#include "nn_models/model.h"






extern int DEBUG_MODE;
extern int VIEW_TENSORS;
extern int MEASURE_MODE;
extern int BENCHMARK_MODEL;
extern int CHECK_INPUT_OUTPUT;


extern int CACHE_ENABLED;
extern int LAYER_WISE_UPDATE_ENABLED;

int global_it;

// How often we update (in micro sec)
//#define UPDATE_PERIOD 10000 //10 ms
//#define UPDATE_PERIOD 1000 //1 ms

//#define UPDATE_PERIOD 4000 //4 ms
//#define UPDATE_PERIOD 0 //0 ms
//#define UPDATE_PERIOD 3000 //3 ms

//#define UPDATE_PERIOD 600 //0.6 ms
#define UPDATE_PERIOD 6000 //6 ms
//#define UPDATE_PERIOD 60000 //60 ms
//#define UPDATE_PERIOD 600000 //600 ms
//#define UPDATE_PERIOD 0 //0





//NNLayer* FC_LIF_Layer_Init(

    //// Const Tensors
    //const uint8_t* command_stream,
    //size_t command_stream_length,

    //int8_t** memory_region_ptrs,
    //size_t* memory_region_sizes,
    //size_t num_regions,

    //int8_t* in_spk,
    //size_t input_layer_size,
    //int8_t* out_spk,
    //size_t output_layer_size,

    //// Non-const tensors

    ////size_t bias_relative_addr,
    ////size_t weight_relative_addr,
    ////size_t v_mem_relative_addr,
    ////size_t time_not_updated_relative_addr,
    ////size_t update_nxt_layer_relative_addr,

    ////size_t v_mem_size,
    ////size_t time_not_updated_size,
    ////size_t update_nxt_layer_size,
    ////size_t bias_tensor_size,
    ////size_t weight_tensor_size,

    //char** tensor_names,
    //size_t* tensor_relative_addrs,
    //size_t* tensor_regions,
    //size_t* tensor_sizes,
    //size_t num_tensors,


    ////float in_spk_scale,
    ////int in_spk_zero_point,

    ////float v_mem_scale,
    ////int v_mem_zero_point,
    ////float time_not_updated_scale,
    ////int time_not_updated_zero_point,

    ////float out_spk_scale,
    ////int out_spk_zero_point
    //int is_last_layer
//) {


    ///* ____________________________________________ */
    ///* 1. Initiate NNLayer Struct                   */

    //NNLayer* nnlayer = NNLayer_Init(num_tensors, num_regions);
    //if (nnlayer == NULL) { printf("Error when initializing NN_layer0\n"); }



    //nnlayer->command_stream = command_stream;
    //nnlayer->command_stream_length = command_stream_length;


   

    //// Assign memory regions
    //for (size_t i = 0; i < num_regions; i++) {
        //nnlayer->memory_regions[i].region_start_ptr = memory_region_ptrs[i];
        //nnlayer->memory_regions[i].length = memory_region_sizes[i];
    //}






    //// Assign Tensors
    //for (size_t i = 0; i < num_tensors; i++) {
        //NNLayer_Assign(layer, );
    //}



    ////3. Assign default values to V_mem
    //quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[V_MEM_QUANT_IDX], output_layer_size, v_mem_scale, v_mem_zero_point);

    //// Assign default value to Time_not_updated
    //quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[TIME_NOT_UPDATED_QUANT_IDX], 1, time_not_updated_scale, time_not_updated_zero_point);

    //// Assign default value to -1
    //nnlayer->time_of_previous_update = -1;



    //// There is an extra tensor out_spk_sum if it is last layer
    //if (is_last_layer) {
        //int8_t* out_spk_sum = PersistentAllocator_GetAbsPointer(&nnlayer->allocator, out_spk_sum_relative_addr);
        //NNLayer_Assign(nnlayer, OUT_SPK_SUM_TENSOR_IDX,  out_spk_sum , output_layer_size, out_spk_sum_scale, out_spk_sum_zero_point, "out_spk_sum");
        //quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[OUT_SPK_SUM_TENSOR_IDX], output_layer_size, out_spk_sum_scale, out_spk_sum_zero_point);
    //}


    //nnlayer->output_size = nnlayer->tensor_sizes[OUT_SPK_TENSOR_IDX];
    //nnlayer->input_size = nnlayer->tensor_sizes[IN_SPK_TENSOR_IDX];

    //// Assign layer input and output (so other layers know where to read and write from)
    //nnlayer->input = in_spk;
    //nnlayer->update_nxt = update_nxt_layer;
    //nnlayer->output = out_spk;




    //return nnlayer;


//}






NN_Model* MLP_Init() {


    // 1. Allocate for Total Arena Tensor on Heap



    // Do this for each layer we have
    // First NNLayer

    printf("SNN Model:\n");

    NNLayer* layer_pointers[MLP_NUM_LAYERS];
    for (size_t layer_num = 0; layer_num < MLP_NUM_LAYERS; layer_num++) {

        printf("Layer: %d================\n\n", layer_num);

        layer_pointers[layer_num] = init_layers_func[layer_num]();

        printf("\tFC_LIF_%d: %d X %d\n", layer_num, layer_pointers[layer_num]->input->size, layer_pointers[layer_num]->output->size);

    }



    // Check Input output addresses
    printf("Try accessing the input outputs");
    for (size_t layer_num=0; layer_num < MLP_NUM_LAYERS; layer_num++) {
        printf("layer: %d\n", layer_num);
        printf("Input:\n");
        Tensor_Print(layer_pointers[layer_num]->input);
        printf("Output:\n");
        Tensor_Print(layer_pointers[layer_num]->output);
    }

    /* For Debugging

    // Try accessing all the memory regions
    printf("Try accessing the memory regions");
    for (size_t layer_num=0; layer_num < MLP_NUM_LAYERS; layer_num++) {
        printf("layer: %d\n", layer_num);
        for (size_t region_num=0;  region_num < layer_pointers[layer_num]->num_regions; region_num++){
            MemoryRegion* mem_reg = layer_pointers[layer_num]->memory_regions[region_num];
            printf("memory_region_num: %d\n", region_num);
            printf("\tptr: %p\n", mem_reg->region_start_ptr);
            printf("\tsize: %d\n", mem_reg->length);
        }
    }

    //Try Accessing all the Tensor Regions
    printf("Try accessing the tensors");
    for (size_t layer_num=0; layer_num < MLP_NUM_LAYERS; layer_num++) {
        printf("layer: %d\n", layer_num);
        for (size_t tensor_num=0;  tensor_num < layer_pointers[layer_num]->num_tensors; tensor_num++){
            Tensor* tmp_tensor = layer_pointers[layer_num]->tensors[tensor_num];
            //printf("tensor_num: %d\n\tptr: %p\n\tsize: %d\n", region_num, mem_reg->region_start_ptr, mem_reg->length);
            Tensor_Print(tmp_tensor);
        }
    }
    */


    for (size_t layer_num = 0; layer_num < MLP_NUM_LAYERS; layer_num++) {

        // Set Next layer (to get linked list)
        if (layer_num < MLP_NUM_LAYERS-1) {
            layer_pointers[layer_num]->next_layer = layer_pointers[layer_num+1];
        } else {
            layer_pointers[layer_num]->next_layer = NULL;
        }


        // Set update curr
        if (layer_num != 0) {
            layer_pointers[layer_num]->update_curr_layer = layer_pointers[layer_num-1]->update_nxt_layer;
        } else {
            layer_pointers[layer_num]->update_curr_layer = NULL;
        }

    }

    


    // 3. Create NN_Model
    NN_Model* mlp_model = NN_Model_Init(NULL, layer_pointers[0], MLP_INPUT_LAYER_SIZE, MLP_OUTPUT_LAYER_SIZE, MLP_NUM_TIME_STEPS);

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



int MLP_Run_Layer(NNLayer* nnlayer)
{


    if (DEBUG_MODE) {

        // print values
        printf("BEFORE INVOKE\n");
        //PrintTensor("tensor_arena", tensor_arena, tensor_arena_size);
    }



    // Assign base addrs
    const size_t num_regions = nnlayer->num_regions;
    uint64_t base_addrs[num_regions];
    size_t base_addrs_size[num_regions];

    for (size_t i = 0; i < num_regions; i++) {
        base_addrs[i] = (uint64_t)(intptr_t)nnlayer->memory_regions[i]->region_start_ptr;
        base_addrs_size[i] = nnlayer->memory_regions[i]->length;
    }

    //base_addrs[0] = (uint64_t)(intptr_t)weight_tensor;   // Model weights
    //base_addrs[1] = (uint64_t)(intptr_t)tensor_arena;   // Tensor arena pointer
    //base_addrs[2] = (uint64_t)(intptr_t)tensor_arena;   // Fast scratch, same as tensor arena for now
    //base_addrs[3] = (uint64_t)(intptr_t)lif_param;
    //base_addrs[4] = (uint64_t)(intptr_t)exp_lut;
    //base_addrs[5] = (uint64_t)(intptr_t)input;
    //base_addrs[6] = (uint64_t)(intptr_t)output;

    //base_addrs_size[0] = weight_tensor_size;
    //base_addrs_size[1] = tensor_arena_size;
    //base_addrs_size[2] = tensor_arena_size;
    //base_addrs_size[3] = lif_param_size;
    //base_addrs_size[4] = exp_lut_size;
    //base_addrs_size[5] = input_size;
    //base_addrs_size[6] = output_size;


    // Sanity check to ensure num_tensors matches length of tensors
    if (sizeof(base_addrs) / sizeof(base_addrs[0]) != num_regions) { printf("num_tensors does not match base_addrs length\n"); return -1;}
    if (sizeof(base_addrs_size) / sizeof(base_addrs_size[0]) != num_regions) { printf("num_tensors does not match base_addrs_size length\n"); return -1;}




    // Run NPU commands
    if(run_cms(nnlayer->command_stream, nnlayer->command_stream_length, base_addrs, base_addrs_size, num_regions) != 0) {
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
        Tensor* v_mem = NNLayer_Get_Tensor(nnlayer, "V_MEM");
        quantize_float_scalar_to_int8_array(0, v_mem->ptr, v_mem->size, v_mem->scale, v_mem->zero_point);
        nnlayer = nnlayer->next_layer;
    }
}


void reset_out_spk(NN_Model* mlp_model) {
    NNLayer* nnlayer = mlp_model->first_nnlayer;

    while(nnlayer != NULL) {

        // Note: out_spk_sum should realistically only exist in the last layer
        Tensor* out_spk_sum = NNLayer_Get_Tensor(nnlayer, "OUT_SPK_SUM");
        quantize_float_scalar_to_int8_array(0, out_spk_sum->ptr, out_spk_sum->size, out_spk_sum->scale, out_spk_sum->zero_point);

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
    int layer_num = 0;
    while(nnlayer != NULL) {
        //printf("layer: %d\n\n", layer_num); layer_num+=1;
        for (size_t i = 0; i < nnlayer->num_tensors; i++) {
            //printf("tensor: %d\n", i);
            /* TEMPORARY SOLUTION !!!! */
            Tensor* tensor = nnlayer->tensors[i];
            //Tensor_Print(tensor);
            // If its V_mem or time not updated, assign 0 to them
            if (strcmp(tensor->name, "V_MEM") == 0 || strcmp(tensor->name, "TIME_NOT_UPDATED") == 0) {
                //printf("v_mem/time_not_updated\n");
                //Tensor_Print(tensor);
                quantize_float_scalar_to_int8_array(TENSOR_INIT_VALUE, tensor->ptr, tensor->size, tensor->scale, tensor->zero_point);
            } else if (strcmp(tensor->name, "IN_SPK") == 0) {  // Assign layer input and output (so other layers know where to read and write from)
                //printf("in_spk\n");
                //Tensor_Print(tensor);
                quantize_float_scalar_to_int8_array(0, tensor->ptr, tensor->size, tensor->scale, tensor->zero_point);
            } else if (strcmp(tensor->name, "OUT_SPK") == 0) {
                //printf("out_spk\n");
                //Tensor_Print(tensor);
                quantize_float_scalar_to_int8_array(0, tensor->ptr, tensor->size, tensor->scale, tensor->zero_point);
            } else if (strcmp(tensor->name, "UPDATE_NXT_LAYER") == 0) {
                //printf("update_nxt_layer\n");
                //Tensor_Print(tensor);
                quantize_float_scalar_to_int8_array(0, tensor->ptr, tensor->size, tensor->scale, tensor->zero_point);
            } else if (strcmp(tensor->name, "OUT_SPK_SUM") == 0 && nnlayer->next_layer == NULL){
                //printf("out_spk_sum\n");
                //Tensor_Print(tensor);
                quantize_float_scalar_to_int8_array(0, tensor->ptr, tensor->size, tensor->scale, tensor->zero_point);
            }
        }
        //// Reset Membrane Potential
        //quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[V_MEM_QUANT_IDX], nnlayer->tensor_sizes[V_MEM_QUANT_IDX], nnlayer->quant_params[V_MEM_QUANT_IDX].scale, nnlayer->quant_params[V_MEM_QUANT_IDX].zero_point);

        //// Reset Out_spk
        //quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[OUT_SPK_TENSOR_IDX], nnlayer->tensor_sizes[OUT_SPK_TENSOR_IDX], nnlayer->quant_params[OUT_SPK_TENSOR_IDX].scale, nnlayer->quant_params[OUT_SPK_TENSOR_IDX].zero_point);

        // Reset time stamp for previous update
        nnlayer->time_of_previous_update = -1;


        //// Reset out_sum_spk (only exists for last layer)
        //if (nnlayer->next_layer == NULL) {
            ////quantize_float_scalar_to_int8_array(0, nnlayer->tensor_ptrs[OUT_SPK_SUM_TENSOR_IDX], nnlayer->tensor_sizes[OUT_SPK_SUM_TENSOR_IDX], nnlayer->quant_params[OUT_SPK_SUM_TENSOR_IDX].scale, nnlayer->quant_params[OUT_SPK_SUM_TENSOR_IDX].zero_point);
        //}

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


// measure inference exe time
extern double ait_mlp_run_layer;




extern double ait_reset_model_for_new_sample;
extern double ait_set_test_pattern_pointer_to_model;
extern double ait_get_time_since_last_update;
extern double ait_ethosu_release_driver;
extern double ait_arg_max;

#include CMSIS_device_header

extern double ait_disable_dcache;
extern double ait_enable_dcache;

int MLP_Inference_test_patterns(
    NN_Model* mlp_model,
    int make_printouts
) {


    // Get the test samples, test targets and number of samples
    volatile int8_t (*test_patterns)[MLP_NUM_TIME_STEPS][MLP_INPUT_LAYER_SIZE] = get_test_inputs();
    volatile int8_t* test_targets = get_test_target();
    size_t num_samples = NUM_TEST_SAMPLES;
    



    // Disable Cache
    uint32_t ait_disable_dcache_start_tick = debug_start_timer();
    if (!CACHE_ENABLED) {
        SCB_DisableDCache();
    }


    ait_disable_dcache += debug_end_timer(ait_disable_dcache_start_tick);



    /* 
    
    Declare Local Variables for Debugging and Benchmarking

    */
    // For debugging
    size_t number_of_no_spk = 0;
    uint32_t debug_timer_start; float debug_timer_elapsed_ms;


    // For setting to WFE__ if inference time < UPDATE_PERIOD
    uint32_t inference_start_tick;
    uint32_t inference_elapsed_ticks;


    // For Benchmarking accuracy
    size_t correct = 0;
    size_t prediction_arr [NUM_TEST_SAMPLES] = { 0 };

    // For benchmarking inference speed
    double average_inference_time = 0;

    double exclude_sleep_time = 0;
    double avg_inference_time_per_sample = 0;
    double avg_inference_time_per_forward_pass = 0;
    int32_t max_inference_time_per_forward_pass = -UPDATE_PERIOD;


    // Percentage of time we spike
    size_t at_least_one_out_spk[MLP_NUM_LAYERS] = { 0 };
    size_t num_spikes_we_had[MLP_NUM_LAYERS] = { 0 };
    size_t total_num_spikes[MLP_NUM_LAYERS] = { 0 };








    float time_steps_layer_not_updated;


    // For every input sample (in real system would be while(true) loop)

    size_t it = 0;
    //global_it = it;     // For debugging
    
    while (it < num_samples) {


        uint32_t inference_speed_measure_each_sample_start_tick = debug_start_timer();



        // Reset Membrane potential between input samples
        // Reset output spike in case last layer is never computed (because no output spikes at all from the layer before that)
        // Reset time stamp for previous layer update
        reset_model_for_new_sample(mlp_model);




        ait_reset_model_for_new_sample += debug_end_timer(inference_speed_measure_each_sample_start_tick);




        // Feed the same input to the network for num_time_steps
        for (size_t time_step = 0; time_step < mlp_model->num_time_steps; time_step++){

            uint32_t avg_inference_time_per_forward_pass_start_tick = debug_start_timer();


            // Set First Layer as current layer
            NNLayer* nnlayer = mlp_model->first_nnlayer;



            /*
            Copy test patterns to input tensor
            In original version (v1.0), it was done by changing the input pointer. Current version cannot do this yet
            Can likely solve this by changing to new value in:
                - NNLayer->input
                - NNLayer->memory_regions
                - NNLayer->tensors
            */
            int8_t* tmp_in_arr = nnlayer->input->ptr;
            for (size_t neuron_num = 0; neuron_num < nnlayer->input->size; neuron_num++) {
                tmp_in_arr[neuron_num] = test_patterns[it][time_step][neuron_num];

            }





            // Iterate over all layers, if they have input spike --> compute, otherwise done with forward pass
            size_t layer_number = 0;
            while (nnlayer != NULL) {
                

                // Had at least 1 spike in layer0 --> run next layer
                if ( nnlayer == mlp_model->first_nnlayer || ((int8_t)(nnlayer->update_curr_layer->ptr[0]) == 127 || !LAYER_WISE_UPDATE_ENABLED) ){

                    // measure inference exe time: start tick
                    uint32_t ait_get_time_since_last_update_start_tick = debug_start_timer();



                    // Update how long it was we updated layer0 last
                    time_steps_layer_not_updated = time_step - nnlayer->time_of_previous_update;
                    //quantize_float_scalar_to_int8_array(time_steps_layer_not_updated, nnlayer->tensor_ptrs[TIME_NOT_UPDATED_QUANT_IDX], nnlayer->tensor_sizes[TIME_NOT_UPDATED_QUANT_IDX], nnlayer->quant_params[TIME_NOT_UPDATED_QUANT_IDX].scale, nnlayer->quant_params[TIME_NOT_UPDATED_QUANT_IDX].zero_point);
                    
                    // Reset TIME_NOT_UPDATED
                    for (size_t i = 0; i < nnlayer->num_tensors; i++ ) {
                        if (strcmp(nnlayer->tensors[i]->name, "TIME_NOT_UPDATED") == 0) {
                            quantize_float_scalar_to_int8_scalar(time_steps_layer_not_updated, nnlayer->tensors[i]->ptr, nnlayer->tensors[i]->scale_reciprocal, nnlayer->tensors[i]->zero_point);
                        }
                    }



                    // measure inferece exe time
                    ait_get_time_since_last_update += debug_end_timer(ait_get_time_since_last_update_start_tick);
                    uint32_t inference_speed_measure_start_tick = debug_start_timer();


                    // Run Layer (sends task to NPU)
                    MLP_Run_Layer(nnlayer);


                    nnlayer->time_of_previous_update = time_step;
                ait_mlp_run_layer += debug_end_timer(inference_speed_measure_start_tick);
        

                // Running this code means that we computed the layer (since otherwise we wouldve break and exit while loop)
                at_least_one_out_spk[layer_number] += 1;
                // Store the spikes received
                for (size_t i = 0; i < nnlayer->output->size; i++) {
                    total_num_spikes[layer_number] += nnlayer->output->ptr[i];
                }
                


                } else if ((int8_t)(nnlayer->update_curr_layer->ptr[0]) == -128) {
                    break;
                } else { //printf("ERROR: Unexpected update_nxt_layer value found. Expected 127 or -128 but received: %d\n", (int8_t)*(nnlayer->update_curr)); 
                }



                

                // To next layer
                nnlayer = nnlayer->next_layer;

                layer_number += 1;


            }
            


            uint32_t inference_time_this_forward_pass = debug_end_timer(avg_inference_time_per_forward_pass_start_tick);

            if ((int32_t)inference_time_this_forward_pass > max_inference_time_per_forward_pass) {
                max_inference_time_per_forward_pass = (int32_t)inference_time_this_forward_pass;
            }
            avg_inference_time_per_forward_pass += (double)inference_time_this_forward_pass;


            // Sleep: Set CPU to WFE state if done and waiting for next inference forward pass cycle
            uint32_t exclude_sleep_time_start_tick = debug_start_timer();
            // Delay before starting next inference cycle (time step)
            uint32_t elapsed_us = debug_end_timer(avg_inference_time_per_forward_pass_start_tick);
            int32_t remaining_us = (int32_t)UPDATE_PERIOD - (uint32_t)elapsed_us; 
            if (remaining_us > 0) { 
                delay(remaining_us);
            }
            exclude_sleep_time += debug_end_timer(exclude_sleep_time_start_tick);

            
        
        }



        // measure inference exe time: start tick
        uint32_t ait_arg_max_start_tick = debug_start_timer();

        // Decode: Decide prediction by checking which output neuron had the most amount of output spikes
        size_t pred = arg_max(mlp_model->out_spk_sum->ptr, mlp_model->out_spk_sum->size, mlp_model->out_spk_sum->scale, mlp_model->out_spk_sum->zero_point);


        //measure inference exe time: capture elapsed time
        ait_arg_max += debug_end_timer(ait_arg_max_start_tick);



        





        // Measure inference time for each sample
        avg_inference_time_per_sample += debug_end_timer(inference_speed_measure_each_sample_start_tick);




        if (make_printouts) {
            // Check if correct or not and add to counter
            if (pred == (size_t)test_targets[it]) { correct += 1; }
            prediction_arr[it] = pred;

        }


        it++;

    }

    uint32_t ait_enable_dcache_start_tick = debug_start_timer();


    if (!CACHE_ENABLED) {
        SCB_EnableDCache();
    }

    ait_enable_dcache += debug_end_timer(ait_enable_dcache_start_tick);







    if (make_printouts) {
        //printf("All Predictions:\n");
        //for (size_t i = 0; i < num_samples; i++) {
            //printf("%d, ", prediction_arr[i]);
        //}
        //printf("\n");
        // Show stats at the end

        printf("Inference period:, %d\n", UPDATE_PERIOD);

        double accuracy = (double)correct / (double)num_samples;
        printf("The total accuracy over %d input patterns is, %f,\n", num_samples, accuracy);
        //printf("Num samples with zero output spikes across all time steps, %d,\n", number_of_no_spk);


        printf("Max inference forward pass time is: %d\n", max_inference_time_per_forward_pass);


        // Print average layer outspk
        printf("Average number of times we have at least one spike at each layer\n");
        printf("Layer number,Percentage of times we update layer,Average Num Spikes Received Per layer update,\n");
        printf("Start printing layer update ratio\n");
        //printf("\tPercentage of times we update layer: %f\n", percentage_of_times_we_update_layer);
        //printf("\tAverage Num Spikes Received Per layer update: %f\n", (double)total_num_spikes[layer] / (double)at_least_one_out_spk[layer]);
        NNLayer* nnlayer_check_update_nxt = mlp_model->first_nnlayer;
        //for (size_t layer = 0; layer < mlp_model->num_layers; layer++) {
        size_t layer = 0;
        while (nnlayer_check_update_nxt != NULL) {
            double percentage_of_times_we_update_layer = ((double)at_least_one_out_spk[layer] / (double)num_samples) / (double) mlp_model->num_time_steps;
            double tmp = (double)total_num_spikes[layer] / (double)nnlayer_check_update_nxt->output->size / (double)at_least_one_out_spk[layer] / (double) mlp_model->num_time_steps;
            double avg_num_spikes_per_layer_update = (double)total_num_spikes[layer] / (double)at_least_one_out_spk[layer];

            //printf("\tPercentage of times we update layer: %f\n", percentage_of_times_we_update_layer);
            //printf("\tAverage Num Spikes Received Per layer update: %f\n", (double)total_num_spikes[layer] / (double)at_least_one_out_spk[layer]);
            //printf("\n");
            
            //printf("%d, %f, %f\n", layer, percentage_of_times_we_update_layer, avg_num_spikes_per_layer_update);
            printf("%f\n", percentage_of_times_we_update_layer);

            nnlayer_check_update_nxt = nnlayer_check_update_nxt->next_layer;
            layer += 1;
        }
        printf("Stop printing layer update ratio\n");


        printf("model name: %s\n", MODEL_NAME);
        printf("\n\n\n\n\n\n\n\n\n\n\n");
        printf("start printing inference exe time\n");
        avg_inference_time_per_forward_pass /= ((double)num_samples * (double)mlp_model->num_time_steps);
        //avg_inference_time_per_sample /= (double)NUM_TEST_SAMPLES;
        double tmp = (avg_inference_time_per_sample - exclude_sleep_time) / (double)num_samples;
        printf("avg_inference_time_per_forward_pass, %f,\n", avg_inference_time_per_forward_pass);
        printf("avg_inference_exe_time_per_sample, %f,\n", tmp);
        printf("stop printing inference exe time\n");

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







