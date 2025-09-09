import math
import os, sys



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ethosu.vela.api import *



from constants import *
from data_structs import MemoryAllocator, Region
from config_ops import *
#from config_ops import gen_weights_and_biases
from extra_func import gen_cms
from write_connectivity_h_file import write_init_func
from write_layer_files import write_cms_to_files

from funcs_for_npu_op_gen import get_int8_fc_weights_and_biases, get_elementwise_op, get_elementwise_op_with_lut, get_pooling_op, get_fully_connected_op, generate_dict_for_writing_defines_to_C_files


def gen_fc_lif(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, 
        weights_volume_ohwi, bias_list, beta_list, vth_list,
        #cms_name,
        layer_base_name,
        layer_num,
        weights_and_biases_on_sram, lif_params_on_sram, is_last_layer, NUM_TIME_STEPS,

        TIME_NOT_UPDATED_MAX_VAL,
        TIME_NOT_UPDATED_MIN_VAL,

        IN_CURR_MAX_VAL,
        IN_CURR_MIN_VAL,

        V_MEM_MAX_VAL,
        V_MEM_MIN_VAL,

        DECAY_ACC_MAX_VAL,
        DECAY_ACC_MIN_VAL,

        DECAY_MAX_VAL,
        DECAY_MIN_VAL,

        DECAYED_MEM_MAX_VAL,
        DECAYED_MEM_MIN_VAL,
         
        DEBUG_MODE, ACCELERATOR,
        mem_store_loc,
        header_out_filepath, connectivity_filepath):



    cms_name = layer_base_name + str(layer_num)

    '''
    Set FM Quantization Params
    '''


    # Must be  >= 0
    MIN_DIFF = 0.1


    # Take the natural log of each value
    ln_beta_values = [math.log(v) for v in beta_list]
    max_ln_beta_value = max(ln_beta_values)
    min_ln_beta_value =  min(ln_beta_values)

    LN_BETA_MAX_VAL = max_ln_beta_value
    LN_BETA_MIN_VAL = min_ln_beta_value

    if (max_ln_beta_value == min_ln_beta_value):
        LN_BETA_MAX_VAL += MIN_DIFF
        LN_BETA_MIN_VAL -= MIN_DIFF
    




    max_vth_value = max(vth_list)
    min_vth_value = min(vth_list)
    VTH_MAX_VAL = max_vth_value
    VTH_MIN_VAL = min_vth_value

    if (min_vth_value == max_vth_value):
        VTH_MAX_VAL += MIN_DIFF
        VTH_MIN_VAL -= MIN_DIFF





    if (is_last_layer):
        # MAX VAL == NUM_TIME_STEPS
        OUT_SPK_SUM_MAX_VAL = NUM_TIME_STEPS
        OUT_SPK_SUM_MIN_VAL = 0



    ###########
    # Autoset Params (depends on the previously set quantization params)

    # Must be same for input and output quantization
    IN_SPK_MAX_VAL = 127
    IN_SPK_MIN_VAL = -128
    OUT_SPK_MAX_VAL = 127
    OUT_SPK_MIN_VAL = -128

    # Only need to differentiate between > 0 and < 0
    V_MEM_SUB_VTH_MAX_VAL = 1
    V_MEM_SUB_VTH_MIN_VAL = -1


    # Reset is either 0 or VTH --> same quantization params as VTH
    RESET_MAX_VAL = VTH_MAX_VAL
    RESET_MIN_VAL = 0


    # Only need to differentiate between 0 and anything else
    UPDATE_NXT_LAYER_MAX_VAL = 1
    UPDATE_NXT_LAYER_MIN_VAL = 0

    ###########



    # Need to make sure that in_spk and out_spk have the same quantization parameters
    if (IN_SPK_MAX_VAL != OUT_SPK_MAX_VAL or IN_SPK_MIN_VAL != OUT_SPK_MIN_VAL):
        print("IN_SPK and OUT_SPK do not match")
        exit()






    '''
    Generate Quantization Parameters from the Value range set above for each tensor
    '''
    IN_SPK_SCALE, IN_SPK_ZERO_POINT = zero_point_quant(IN_SPK_MAX_VAL, IN_SPK_MIN_VAL)


    # Layer params
    LN_BETA_SCALE, LN_BETA_ZERO_POINT = zero_point_quant(LN_BETA_MAX_VAL, LN_BETA_MIN_VAL)
    VTH_SCALE, VTH_ZERO_POINT = zero_point_quant(VTH_MAX_VAL, VTH_MIN_VAL)

    # TMP Feature maps
    IN_CURR_SCALE, IN_CURR_ZERO_POINT = zero_point_quant(IN_CURR_MAX_VAL, IN_CURR_MIN_VAL)




    '''
    Quantize and Decrypt Weight, bias and Time constant (beta), and Vth
    '''
    # Generate Weights and Bias list
    weight_byte_arr_init, bias_byte_arr_init = get_int8_fc_weights_and_biases(weights_volume_ohwi, bias_list, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, IN_SPK_SCALE, IN_CURR_SCALE, ACCELERATOR, DEBUG_MODE)


    # Generate LIF Params Quant List
    LN_BETA_QUANT_LIST = generate_ln_beta_values(beta_list=beta_list, ln_beta_scale=LN_BETA_SCALE, ln_beta_zero_point=LN_BETA_ZERO_POINT)    
    VTH_QUANT_LIST = quantize_vth_values(vth_list=vth_list, vth_scale=VTH_SCALE, vth_zero_point=VTH_ZERO_POINT)



    '''
    
    Assign Memory segments in SRAM Scratch (region 1)

    '''
    mem_alloc = MemoryAllocator(INPUT_LAYER_SIZE,
                                OUTPUT_LAYER_SIZE,
                                {
                                "PARAMS_REGION" : Region(3, "PARAMS_REGIONS", "const int8_t", True),
                                "LUT_REGION" : Region(4, "LUT_REGION", "const int8_t", True),
                                #INPUT_REGION_NAME : Region(5, "int8_t", False),
                                #OUTPUT_REGION_NAME : Region(6, "int8_t", False),
                                }
                                )



    mem_alloc.alloc(SRAM_SCRATCH_REGION_NAME, "TMP1", OUTPUT_LAYER_SIZE, NpuLayout.NHCWB16)
    mem_alloc.alloc(SRAM_SCRATCH_REGION_NAME, "TMP2", OUTPUT_LAYER_SIZE, NpuLayout.NHCWB16)
    mem_alloc.alloc(SRAM_SCRATCH_REGION_NAME, "V_MEM", OUTPUT_LAYER_SIZE, NpuLayout.NHCWB16)

    if is_last_layer:
        mem_alloc.alloc(SRAM_SCRATCH_REGION_NAME, "OUT_SPK_SUM", OUTPUT_LAYER_SIZE, NpuLayout.NHCWB16)

    mem_alloc.alloc(SRAM_SCRATCH_REGION_NAME, "TIME_NOT_UPDATED", 1)
    mem_alloc.alloc(SRAM_SCRATCH_REGION_NAME, "UPDATE_NXT_LAYER", 1)

      
    # WEIGHTS AND BIAS REGION
    mem_alloc.alloc(MODEL_REGION_NAME, "BIAS", len(bias_byte_arr_init))
    mem_alloc.alloc(MODEL_REGION_NAME, "WEIGHT", len(weight_byte_arr_init))


    #if "OUT_SPK_SUM" in mem_alloc.regions[SRAM_SCRATCH_REGION_NAME].memory_map:
        #OUT_SPK_SUM_ADDR = mem_alloc.regions[SRAM_SCRATCH_REGION_NAME].memory_map["OUT_SPK_SUM"]
    #else:
        #OUT_SPK_SUM_ADDR = -1




    # Assign Memory segments for region 3
    mem_alloc.alloc("PARAMS_REGION", "LN_BETA", OUTPUT_LAYER_SIZE)
    mem_alloc.alloc("PARAMS_REGION", "VTH", OUTPUT_LAYER_SIZE)


    # Assign Memory segments for region 4
    DECAY_LUT_INDEX = 0
    CHECK_SPK_LUT_INDEX = 1



    # Assign Memory segment for region 5
    mem_alloc.alloc(INPUT_REGION_NAME, "IN_SPK", INPUT_LAYER_SIZE)

    # Assign Memory segment for region 6
    mem_alloc.alloc(OUTPUT_REGION_NAME, "OUT_SPK", OUTPUT_LAYER_SIZE)



    # Print the allocated memory for debugging
    print("Layer", layer_base_name+str(layer_num))
    mem_alloc.print_mem_regions()

    

    '''

    Define tensors here!!!

    '''

    ln_beta_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="PARAMS_REGION",
        segment_name="LN_BETA",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=LN_BETA_MAX_VAL,
        min_fm_value=LN_BETA_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="LN_BETA"
    )


    time_not_updated_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=1,
        region_name=SRAM_SCRATCH_REGION_NAME,
        segment_name="TIME_NOT_UPDATED",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=TIME_NOT_UPDATED_MAX_VAL,
        min_fm_value=TIME_NOT_UPDATED_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="TIME_NOT_UPDATED"
    )

    decay_acc_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name=SRAM_SCRATCH_REGION_NAME,
        segment_name="TMP1",
        layout=NpuLayout.NHCWB16,
        data_type=NpuDataType.INT8,
        max_fm_value=DECAY_ACC_MAX_VAL,
        min_fm_value=DECAY_ACC_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="DECAY_ACC"
    )

    in_spk_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=INPUT_LAYER_SIZE,
        region_name=INPUT_REGION_NAME,
        segment_name="IN_SPK",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=IN_SPK_MAX_VAL,
        min_fm_value=IN_SPK_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="IN_SPK"
    )
    

    weight_tensor = mem_alloc.create_weight_and_bias_fm(
        height=1, width=1, depth=len(weight_byte_arr_init),
        region_name=MODEL_REGION_NAME,
        segment_name="WEIGHT",
        data_type=NpuDataType.INT8,
        tensor_values=weights_volume_ohwi,
        tensor_name="WEIGHT"
    )

    
    bias_tensor = mem_alloc.create_weight_and_bias_fm(
        height=1, width=1, depth=len(bias_byte_arr_init),
        region_name=MODEL_REGION_NAME,
        segment_name="BIAS",
        data_type=NpuDataType.INT8,
        tensor_values=bias_list,
        tensor_name="BIAS"
    )
        
        


    in_curr_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name=SRAM_SCRATCH_REGION_NAME,
        segment_name="TMP2",
        layout=NpuLayout.NHCWB16,
        data_type=NpuDataType.INT8,
        max_fm_value=IN_CURR_MAX_VAL,
        min_fm_value=IN_CURR_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="IN_CURR"
    )

    v_mem_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name=SRAM_SCRATCH_REGION_NAME,
        segment_name="V_MEM",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=V_MEM_MAX_VAL,
        min_fm_value=V_MEM_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="V_MEM"
    )

    decay_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name=SRAM_SCRATCH_REGION_NAME,
        segment_name="TMP1",
        layout=NpuLayout.NHCWB16,
        #layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=DECAY_MAX_VAL,
        min_fm_value=DECAY_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="DECAY"
    )



    decayed_mem_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name=SRAM_SCRATCH_REGION_NAME,
        segment_name="TMP1",
        layout=NpuLayout.NHCWB16,
        #layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=DECAYED_MEM_MAX_VAL,
        min_fm_value=DECAYED_MEM_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="DECAYED_MEM"
    )


    vth_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="PARAMS_REGION",
        segment_name="VTH",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=VTH_MAX_VAL,
        min_fm_value=VTH_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="VTH"
    )


    v_mem_sub_vth_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name=OUTPUT_REGION_NAME,
        segment_name="OUT_SPK",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=V_MEM_SUB_VTH_MAX_VAL,
        min_fm_value=V_MEM_SUB_VTH_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="v_mem_sub_vth"
    )


    out_spk_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name=OUTPUT_REGION_NAME,
        segment_name="OUT_SPK",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=OUT_SPK_MAX_VAL,
        min_fm_value=OUT_SPK_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="OUT_SPK"
    )



    reset_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name=SRAM_SCRATCH_REGION_NAME,
        segment_name="TMP2",
        layout=NpuLayout.NHCWB16,
        #layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=RESET_MAX_VAL,
        min_fm_value=RESET_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="RESET"
    )


    update_nxt_layer_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=1,
        region_name=SRAM_SCRATCH_REGION_NAME,
        segment_name="UPDATE_NXT_LAYER",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=UPDATE_NXT_LAYER_MAX_VAL,
        min_fm_value=UPDATE_NXT_LAYER_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="UPDATE_NXT_LAYER"
    )
    


    if is_last_layer:
        out_spk_sum_fm = mem_alloc.create_feature_map_v2(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region_name=SRAM_SCRATCH_REGION_NAME,
            segment_name="OUT_SPK_SUM",
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            max_fm_value=OUT_SPK_SUM_MAX_VAL,
            min_fm_value=OUT_SPK_SUM_MIN_VAL,
            is_symmetric_quant=False,
            tensor_name="OUT_SPK_SUM"
        )






    # Set input and output of the NPU Operation Chain
    mem_alloc.set_input_tensor("IN_SPK")
    mem_alloc.set_output_tensor("OUT_SPK")






    '''
    
    Define the individual NPU Operations here!

    '''

    dma_lut_op, exp_mul_lnb_time_op, decay_lut_values, decay_lut_index = get_elementwise_op_with_lut(NpuElementWiseOp.MUL, ln_beta_fm, time_not_updated_fm, decay_acc_fm,
                                                                                                        decay_fm, 
                                                                                                        mem_alloc, 
                                                                                                        DECAY_LUT_INDEX, math.exp, "LUT_REGION",
                                                                                                        ACCELERATOR)


    mul_decay_op = get_elementwise_op(NpuElementWiseOp.MUL, v_mem_fm, decay_fm, decayed_mem_fm, ACCELERATOR)

    add_decayed_mem_in_curr = get_elementwise_op(NpuElementWiseOp.ADD, decayed_mem_fm, in_curr_fm, v_mem_fm, ACCELERATOR)


    def check_positive(x_real):
        if x_real > 0: y_real = 1
        else: y_real = 0
        return y_real

    check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, check_spk_lut_values, check_spk_lut_index = get_elementwise_op_with_lut(NpuElementWiseOp.SUB, v_mem_fm, vth_fm, v_mem_sub_vth_fm,
                                                                                                                                    out_spk_fm,
                                                                                                                                    mem_alloc,
                                                                                                                                    CHECK_SPK_LUT_INDEX,
                                                                                                                                    check_positive,
                                                                                                                                    "LUT_REGION",
                                                                                                                                    ACCELERATOR)

    reset_mul_vth_out_spk_op = get_elementwise_op(NpuElementWiseOp.MUL, vth_fm, out_spk_fm, reset_fm, ACCELERATOR)

    sub_v_mem_reset_op = get_elementwise_op(NpuElementWiseOp.SUB, v_mem_fm, reset_fm, v_mem_fm, ACCELERATOR)

    update_nxt_layer_reduce_sum_out_spk = get_pooling_op(NpuPoolingOp.REDUCE_SUM, out_spk_fm, update_nxt_layer_fm, kernel_size=(1, 1), accelerator=ACCELERATOR)

    reset_time_op = get_elementwise_op(NpuElementWiseOp.MUL, time_not_updated_fm, None, time_not_updated_fm, ACCELERATOR, ifm2_scalar=0)





    fully_connected_op, weight_byte_arr, bias_byte_arr = get_fully_connected_op(in_spk_fm, in_curr_fm, 
                                                                                weight_tensor, bias_tensor,
                                                                                #weights_volume_ohwi, bias_list, 
                                                                                mem_alloc, 
                                                                                #MODEL_REGION_NAME, "WEIGHT",
                                                                                #MODEL_REGION_NAME, "BIAS",
                                                                                ACCELERATOR)
        

    # Decide the order of the NPU operations
    npu_op_list = [dma_lut_op, exp_mul_lnb_time_op, fully_connected_op, mul_decay_op, add_decayed_mem_in_curr, check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, reset_mul_vth_out_spk_op, sub_v_mem_reset_op, update_nxt_layer_reduce_sum_out_spk, reset_time_op]



    if (is_last_layer):
        incr_out_spk_sum_op = get_elementwise_op(NpuElementWiseOp.ADD, out_spk_sum_fm,  out_spk_fm, out_spk_sum_fm, ACCELERATOR)
        npu_op_list.append(incr_out_spk_sum_op)




    '''
    Wrap up and prepare for writing to C header files
    '''

    # Merge
    lut_arr_contents_str = merge_lut_values_to_str([(decay_lut_values, decay_lut_index), (check_spk_lut_values, check_spk_lut_index)])
    lif_params_arr_contents_str = merge_lif_params_to_str(LN_BETA_QUANT_LIST, VTH_QUANT_LIST)
    weights_and_biases_arr_contents_str = merge_weights_and_biases_to_str(weight_byte_arr, bias_byte_arr)

    cms_bytearr, register_cms = gen_cms(npu_op_list, ACCELERATOR, DEBUG_MODE)

    mem_alloc.regions["LUT_REGION"].arr_values_str = lut_arr_contents_str
    mem_alloc.regions["PARAMS_REGION"].arr_values_str = lif_params_arr_contents_str
    mem_alloc.regions[MODEL_REGION_NAME].arr_values_str = weights_and_biases_arr_contents_str

    # Generate Dicts for writing to C
    sizes_dict, addr_dict, quant_param_dict = generate_dict_for_writing_defines_to_C_files(cms_name=cms_name, mem_alloc=mem_alloc, is_last_layer=is_last_layer,
                                                                                            input_size=INPUT_LAYER_SIZE, output_size=OUTPUT_LAYER_SIZE,
                                                                                            weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)
    # Write layer file
    write_cms_to_files(header_out_filepath, mem_alloc, cms_bytearr, register_cms, mem_store_loc, cms_name, layer_num, sizes_dict, addr_dict, quant_param_dict)


    # Write to connectivity file
    write_init_func(connectivity_filepath, mem_alloc, layer_base_name, layer_num)
    




