from ethosu.vela.api import *




import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_ops import *
from extra_func import *


import numpy as np



def main(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, cms_name, header_out_filepath):

    
    print("INPUT_LAYER_SIZE", INPUT_LAYER_SIZE)
    print("OUTPUT_LAYER_SIZE:", OUTPUT_LAYER_SIZE)

    DEBUG_MODE = False




    ACCELERATOR = NpuAccelerator.Ethos_U55_256



    # Assign Memory Regions (0 - 7)

    WEIGHT_AND_BIASES_REGION = 0
    SRAM_SCRATCH_REGION = 1
    PARAMS_REGION = 3
    LUT_REGION = 4
    INPUT_REGION = 5
    OUTPUT_REGION = 6








    # Set FM Quantization Params

    IN_SPK_MAX_VAL = 127
    IN_SPK_MIN_VAL = -128

    #Must be symmetric
    WEIGHT_MAX_VAL = 127/100
    WEIGHT_MIN_VAL = -128/100

    LN_BETA_MAX_VAL = 0
    LN_BETA_MIN_VAL = -5

    TIME_NOT_UPDATED_MAX_VAL = 16
    TIME_NOT_UPDATED_MIN_VAL = 0

    IN_CURR_MAX_VAL = 3
    IN_CURR_MIN_VAL = 0

    V_MEM_MAX_VAL = 4
    V_MEM_MIN_VAL = 0

    DECAY_ACC_MAX_VAL = 0
    DECAY_ACC_MIN_VAL = -5
    DECAY_MAX_VAL = 1
    DECAY_MIN_VAL = 0

    DECAYED_MEM_MAX_VAL = 1
    DECAYED_MEM_MIN_VAL = 0

    VTH_MAX_VAL = 3
    VTH_MIN_VAL = 0.5




    OUT_SPK_MAX_VAL = 127
    OUT_SPK_MIN_VAL = -128


    ###########
    # Autoset Params (depends on the previously set quantization params)


    # Reset is either 0 or VTH --> same quantization params as VTH
    RESET_MAX_VAL = VTH_MAX_VAL
    RESET_MIN_VAL = 0


    # Only need to differentiate between 0 and anything else
    UPDATE_NXT_LAYER_MAX_VAL = 1
    UPDATE_NXT_LAYER_MIN_VAL = 0

    ###########











    IN_SPK_SCALE, IN_SPK_ZERO_POINT = zero_point_quant(IN_SPK_MAX_VAL, IN_SPK_MIN_VAL)


    # Layer params
    LN_BETA_SCALE, LN_BETA_ZERO_POINT = zero_point_quant(LN_BETA_MAX_VAL, LN_BETA_MIN_VAL)
    VTH_SCALE, VTH_ZERO_POINT = zero_point_quant(VTH_MAX_VAL, VTH_MIN_VAL)

    # Layer status
    V_MEM_SCALE, V_MEM_ZERO_POINT = zero_point_quant(V_MEM_MAX_VAL, V_MEM_MIN_VAL)
    TIME_NOT_UPDATED_SCALE, TIME_NOT_UPDATED_ZERO_POINT = zero_point_quant(TIME_NOT_UPDATED_MAX_VAL, TIME_NOT_UPDATED_MIN_VAL)

    # TMP Feature maps
    DECAY_SCALE, DECAY_ZERO_POINT = zero_point_quant(DECAY_MAX_VAL, DECAY_MIN_VAL)
    DECAY_ACC_SCALE, DECAY_ACC_ZERO_POINT = zero_point_quant(DECAY_ACC_MAX_VAL, DECAY_ACC_MIN_VAL)
    IN_CURR_SCALE, IN_CURR_ZERO_POINT = zero_point_quant(IN_CURR_MAX_VAL, IN_CURR_MIN_VAL)
    DECAYED_MEM_SCALE, DECAYED_MEM_ZERO_POINT = zero_point_quant(DECAYED_MEM_MAX_VAL, DECAYED_MEM_MIN_VAL)


    RESET_SCALE, RESET_ZERO_POINT = zero_point_quant(RESET_MAX_VAL, RESET_MIN_VAL)

    WEIGHT_SCALE, WEIGHT_ZERO_POINT = symmetric_zero_point_quant(WEIGHT_MAX_VAL, WEIGHT_MIN_VAL)
    BIAS_SCALE, BIAS_ZERO_POINT = IN_SPK_SCALE*WEIGHT_SCALE/IN_CURR_SCALE, 0

    # Output Feature Map
    UPDATE_NXT_LAYER_SCALE, UPDATE_NXT_LAYER_ZERO_POINT = zero_point_quant(UPDATE_NXT_LAYER_MAX_VAL, UPDATE_NXT_LAYER_MIN_VAL)
    OUT_SPK_SCALE, OUT_SPK_ZERO_POINT = zero_point_quant(OUT_SPK_MAX_VAL, OUT_SPK_MIN_VAL)




    # Define Weights
    ALL_WEIGHT_VALUES = 0.1
    ALL_BIAS_VALUES = 0

    weights_volume_ohwi = ALL_WEIGHT_VALUES * np.ones((OUTPUT_LAYER_SIZE, 1, 1, INPUT_LAYER_SIZE))

    #Biases
    bias_list = []
    for i in range(OUTPUT_LAYER_SIZE):
    #    #bias_list.append(np.int64(i%4))
        bias_list.append(np.int64(ALL_BIAS_VALUES))

    weight_byte_arr_init, bias_byte_arr_init = get_int8_fc_weights_and_biases(weights_volume_ohwi, bias_list, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, WEIGHT_SCALE, WEIGHT_ZERO_POINT, IN_SPK_SCALE, IN_CURR_SCALE, ACCELERATOR, DEBUG_MODE)




    # Assign Memory segments in SRAM Scratch (region 1)
		
    BIAS_ADDR           	=   0                       +   INPUT_LAYER_SIZE
    WEIGHT_ADDR         	=	BIAS_ADDR               +   len(bias_byte_arr_init) # Bias len
    TMP1_ADDR           	=	WEIGHT_ADDR             +   len(weight_byte_arr_init) #weight len
    TMP2_ADDR           	=	TMP1_ADDR               +   OUTPUT_LAYER_SIZE
    V_MEM_ADDR          	=	TMP2_ADDR               +   OUTPUT_LAYER_SIZE  
    TIME_NOT_UPDATED_ADDR	=	V_MEM_ADDR              +   OUTPUT_LAYER_SIZE
    UPDATE_NXT_LAYER_ADDR	=	TIME_NOT_UPDATED_ADDR   +   1


    TENSOR_ARENA_SIZE	=	INPUT_LAYER_SIZE + 3*OUTPUT_LAYER_SIZE + len(bias_byte_arr_init) +len(weight_byte_arr_init) + 16



    ##############
    # Assign Tmp tensors here!
    DECAY_ADDR = TMP1_ADDR
    IN_CURR_ADDR = TMP2_ADDR
    DECAYED_MEM_ADDR = TMP1_ADDR
    RESET_ADDR = TMP2_ADDR

    ##############



    # Assign Memory segments for region 3
    LN_BETA_ADDR = 0
    VTH_ADDR = LN_BETA_ADDR + OUTPUT_LAYER_SIZE


    # Assign Memory segments for region 4

    DECAY_LUT_INDEX = 0
    CHECK_SPK_LUT_INDEX = 1



    # Assign Memory segment for region 5
    IN_SPK_ADDR = 0

    # Assign Memory segment for region 6
    OUT_SPK_ADDR = 0






    ##### Set LIF Param values #######

    # Generate Beta values
    beta_list = []
    for i in range(OUTPUT_LAYER_SIZE):
        beta_list.append(0.9)

    LN_BETA_QUANT_LIST = generate_ln_beta_values(beta_list=beta_list, ln_beta_scale=LN_BETA_SCALE, ln_beta_zero_point=LN_BETA_ZERO_POINT)

    # Generate Vth values
    vth_list = []
    for i in range(OUTPUT_LAYER_SIZE):
        vth_list.append(1)
    
    VTH_QUANT_LIST = quantize_vth_values(vth_list=vth_list, vth_scale=VTH_SCALE, vth_zero_point=VTH_ZERO_POINT)






    # Create Dicts for writing to C files
    def generate_dict_for_writing_defines_to_C_files(cms_name, weight_byte_arr, bias_byte_arr):


        sizes_dict = {


            cms_name.upper()+"_TENSOR_ARENA_SIZE "  : TENSOR_ARENA_SIZE,
            cms_name.upper()+"_INPUT_LAYER_SIZE "   : INPUT_LAYER_SIZE,             
            cms_name.upper()+"_OUTPUT_LAYER_SIZE "  : OUTPUT_LAYER_SIZE,

            cms_name.upper()+"_WEIGHT_LEN" : len(weight_byte_arr),
            cms_name.upper()+"_BIAS_LEN" : len(bias_byte_arr)        
        }

        addr_dict = {
            # Input Feature map
            cms_name.upper()+"_IN_SPK_ADDR" : IN_SPK_ADDR,

            # Layer params
            cms_name.upper()+"_BIAS_ADDR" : BIAS_ADDR,
            cms_name.upper()+"_WEIGHT_ADDR" : WEIGHT_ADDR,
            cms_name.upper()+"_LN_BETA_ADDR" : LN_BETA_ADDR,
            cms_name.upper()+"_VTH_ADDR" : VTH_ADDR,

            # Layer status
            cms_name.upper()+"_V_MEM_ADDR" : V_MEM_ADDR,
            cms_name.upper()+"_TIME_NOT_UPDATED_ADDR" : TIME_NOT_UPDATED_ADDR,

            # TMP Feature maps
            cms_name.upper()+"_IN_CURR_ADDR" : IN_CURR_ADDR,
            cms_name.upper()+"_DECAY_ADDR" : DECAY_ADDR,
            cms_name.upper()+"_DECAYED_MEM_ADDR" : DECAYED_MEM_ADDR,


            # Output Feature Map
            cms_name.upper()+"_UPDATE_NXT_LAYER_ADDR" : UPDATE_NXT_LAYER_ADDR,
            cms_name.upper()+"_OUT_SPK_ADDR" : OUT_SPK_ADDR

        }




        quant_param_dict = {
            cms_name.upper()+"_IN_SPK_SCALE" : IN_SPK_SCALE,
            cms_name.upper()+"_IN_SPK_ZERO_POINT" : IN_SPK_ZERO_POINT,


            cms_name.upper()+"_BIAS_SCALE" : BIAS_SCALE,
            cms_name.upper()+"_BIAS_ZERO_POINT" : BIAS_ZERO_POINT,
            cms_name.upper()+"_WEIGHT_SCALE" : WEIGHT_SCALE,
            cms_name.upper()+"_WEIGHT_ZERO_POINT" : WEIGHT_ZERO_POINT,


            cms_name.upper()+"_LN_BETA_SCALE" : LN_BETA_SCALE,
            cms_name.upper()+"_LN_BETA_ZERO_POINT" : LN_BETA_ZERO_POINT,
            cms_name.upper()+"_VTH_SCALE" : VTH_SCALE,
            cms_name.upper()+"_VTH_ZERO_POINT" : VTH_ZERO_POINT,
            cms_name.upper()+"_V_MEM_SCALE" : V_MEM_SCALE,
            cms_name.upper()+"_V_MEM_ZERO_POINT" : V_MEM_ZERO_POINT,
            cms_name.upper()+"_TIME_NOT_UPDATED_SCALE" : TIME_NOT_UPDATED_SCALE,
            cms_name.upper()+"_TIME_NOT_UPDATED_ZERO_POINT" : TIME_NOT_UPDATED_ZERO_POINT,

            cms_name.upper()+"_DECAY_SCALE" : DECAY_SCALE,
            cms_name.upper()+"_DECAY_ZERO_POINT" : DECAY_ZERO_POINT,
            cms_name.upper()+"_IN_CURR_SCALE" : IN_CURR_SCALE,
            cms_name.upper()+"_IN_CURR_ZERO_POINT" : IN_CURR_ZERO_POINT,
            cms_name.upper()+"_DECAYED_MEM_SCALE" : DECAYED_MEM_SCALE,
            cms_name.upper()+"_DECAYED_MEM_ZERO_POINT" : DECAYED_MEM_ZERO_POINT,


            # Output
            cms_name.upper()+"_UPDATE_NXT_LAYER_SCALE" : UPDATE_NXT_LAYER_SCALE,
            cms_name.upper()+"_UPDATE_NXT_LAYER_ZERO_POINT" : UPDATE_NXT_LAYER_ZERO_POINT,
            cms_name.upper()+"_OUT_SPK_SCALE" : OUT_SPK_SCALE,
            cms_name.upper()+"_OUT_SPK_ZERO_POINT" : OUT_SPK_ZERO_POINT,
        }

        return sizes_dict, addr_dict, quant_param_dict






    def def_decay_lut():

        IFM2_IS_FIRST_OPERAND = False

        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=PARAMS_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=LN_BETA_ADDR,
            scale=LN_BETA_SCALE,
            zero_point=LN_BETA_ZERO_POINT,
            name="ln_beta"
        )


        ifm2 = create_feature_map(
            height=1, width=1, depth=1,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=TIME_NOT_UPDATED_ADDR,
            scale=TIME_NOT_UPDATED_SCALE,
            zero_point=TIME_NOT_UPDATED_ZERO_POINT,
            name="time_not_updated"
        )


        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            #layout=NpuLayout.NHCWB16,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=DECAY_ADDR,
            scale=DECAY_ACC_SCALE,
            zero_point=DECAY_ACC_ZERO_POINT,
            name="decay"
        )





        #DMA for LUT



        # Handle LUT generation and DMA
        decay_lut_index = DECAY_LUT_INDEX

        import math
        dma_lut_op, decay_lut_values = create_lut_and_dma(approximated_func=math.exp, lut_index=decay_lut_index, lut_region=LUT_REGION, data_type=ofm.data_type, 
                        scale_pre_lut=DECAY_ACC_SCALE, zero_point_pre_lut=DECAY_ACC_ZERO_POINT,
                        scale_post_lut=DECAY_SCALE, zero_point_post_lut=DECAY_ZERO_POINT,
                        accelerator=ACCELERATOR,
                        debug_mode=DEBUG_MODE
        )

        activation = create_activation(
            activation_op=NpuActivationOp.TABLE_LOOKUP,
            min_val=None,
            max_val=None,
            lookup_table_index=decay_lut_index
        )


        exp_mul_lnb_time_op = NpuElementWiseOperation(NpuElementWiseOp.MUL)
    
        #elementwise operation
        exp_mul_lnb_time_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        exp_mul_lnb_time_op.rescale = None

        #NpuBlockOperation
        exp_mul_lnb_time_op.ifm = ifm
        exp_mul_lnb_time_op.ifm2 = ifm2
        exp_mul_lnb_time_op.ifm2_scalar = None   #set if ifm2 is a scalar
        exp_mul_lnb_time_op.ofm = ofm
        exp_mul_lnb_time_op.kernel = None
        exp_mul_lnb_time_op.weights = []
        exp_mul_lnb_time_op.biases = []
        exp_mul_lnb_time_op.padding = None
        exp_mul_lnb_time_op.activation = activation
    
        block_config = get_block_config(exp_mul_lnb_time_op, ACCELERATOR)
        exp_mul_lnb_time_op.block_config = block_config
        exp_mul_lnb_time_op.rounding_mode = NpuRoundingMode.TFL
        exp_mul_lnb_time_op.fused_quantize = False
        exp_mul_lnb_time_op.ifm_upscale = NpuResamplingMode.NONE
        exp_mul_lnb_time_op.accumulator_type = NpuAccumulatorType.Default

        #check_block_config_legal(block_config, exp_mul_lnb_time_op, ACCELERATOR)

        return dma_lut_op, exp_mul_lnb_time_op, decay_lut_values, decay_lut_index


    def def_fullyconnected(IN_SPK_ADDR, IN_CURR_ADDR):




        ifm = create_feature_map(
        height=1, width=1, depth=INPUT_LAYER_SIZE,
        region=INPUT_REGION,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=IN_SPK_ADDR,
        scale = IN_SPK_SCALE,
        zero_point = IN_SPK_ZERO_POINT,
        name="in_spk"
        )



        ifm2 = None


        #ofm = create_feature_map(
            #height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            #region=OUTPUT_REGION,
            ##layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            #data_type=NpuDataType.INT8,
            #fm_elem_size=1,
            #fm_addr=OUT_SPK_ADDR,
            #scale = OUT_SPK_SCALE,
            #zero_point = OUT_SPK_ZERO_POINT,
            #name="in_curr"
        #)

        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            #layout=NpuLayout.NHCWB16,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=IN_CURR_ADDR,
            scale = IN_CURR_SCALE,
            zero_point = IN_CURR_ZERO_POINT,
            name="in_curr"
        )


        # Kernel
        kernel = NpuKernel(
            w=1, h=1, 
            stride_x=1, stride_y=1, dilation_x=1, dilation_y=1
        )


        my_op = NpuConv2DOperation()
        my_op.ifm               =   ifm
        my_op.ifm2              =   ifm2
        my_op.ifm2_scalar       =   None
        my_op.ofm               =   ofm
        block_config = get_block_config(my_op, ACCELERATOR)
    

        block_traversal = NpuBlockTraversal.DEPTH_FIRST


        # Define Weights
        weights_volume_ohwi = ALL_WEIGHT_VALUES * np.ones((ofm.shape.depth, kernel.height, kernel.width, ifm.shape.depth))
        if ifm.data_type == NpuDataType.INT8:
            weight_ifm_bitdepth = 8 #int8
        elif ifm.data_type == NpuDataType.INT16:
            weight_ifm_bitdepth = 16 #int16


        #Biases
        bias_list = []
        for i in range(ofm.shape.depth):
        #    #bias_list.append(np.int64(i%4))
            bias_list.append(np.int64(ALL_BIAS_VALUES))




        weight_byte_arr, bias_byte_arr = gen_weights_and_biases(accelerator=ACCELERATOR,
                                weights_volume_ohwi=weights_volume_ohwi,
                                dilation_xy=(1,1),
                                ifm_bitdepth=weight_ifm_bitdepth,
                                ofm_block_depth=block_config[2],
                                op_type=NpuOperationType.Conv2D,
                                block_traversal=block_traversal,

                                #ONLY FOR 1 DIM FMs!!!!
                                bias_list=bias_list,

                                ifm_scale=ifm.quantization.scale_f32,
                                weight_scale=WEIGHT_SCALE,
                                weight_zero_point=WEIGHT_ZERO_POINT,
                                ofm_scale=ofm.quantization.scale_f32,


                                is_debug_mode=DEBUG_MODE
        )

        weight_n_bias_len = len(bias_byte_arr) + len(weight_byte_arr)
        if DEBUG_MODE:
            print("weight_n_bias_len", weight_n_bias_len)
            print("\tbias_len:", len(bias_byte_arr))
            print("\tweight_len", len(weight_byte_arr))

        

        # Make sure that init is the same as current weights
        if (weight_byte_arr != weight_byte_arr_init):
            print("Error: weight_byte_arr != weight_byte_arr_init", len(weight_byte_arr), "!=", len(weight_byte_arr_init))
            sys.exit(1)
        if (bias_byte_arr != bias_byte_arr_init):
            print("Error: bias_byte_arr != bias_byte_arr_init", len(bias_byte_arr), "!=", len(bias_byte_arr_init))
            sys.exit(1)

    

        


        #BIAS_ADDR = WEIGHT_N_BIAS_ADDR
        #WEIGHT_ADDR = BIAS_ADDR + len(bias_byte_arr)
    
    
        WEIGHT_N_BIAS_ADDR = BIAS_ADDR #Bias before weights

        #DMA    
        dma_src = NpuAddressRange(region=WEIGHT_AND_BIASES_REGION, address=0, length=weight_n_bias_len)
        dma_dst = NpuAddressRange(region=SRAM_SCRATCH_REGION, address=WEIGHT_N_BIAS_ADDR, length=weight_n_bias_len)
        dma_op = NpuDmaOperation(src=dma_src, dest=dma_dst)





    

        padding = NpuPadding(top=0, left=0, bottom=0, right=0)


        weights = NpuAddressRange(region=SRAM_SCRATCH_REGION, address=WEIGHT_ADDR, length=len(weight_byte_arr))
        biases = NpuAddressRange(region=SRAM_SCRATCH_REGION, address=BIAS_ADDR, length=len(bias_byte_arr))

    


        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )

    


        fused_quantize = False





        my_op.kernel            =   kernel
        my_op.weights           =   [weights]
        my_op.biases            =   [biases]
        my_op.padding           =   padding
        my_op.activation        =   activation

        my_op.block_config      =   block_config
        my_op.rounding_mode     =   NpuRoundingMode.TFL
        my_op.fused_quantize    =   fused_quantize
        my_op.ifm_upscale       =   NpuResamplingMode.NONE
        my_op.accumulator_type  =   NpuAccumulatorType.Int32
        my_op.block_traversal   =   block_traversal


        #check_block_config_legal(block_config, my_op, ACCELERATOR)





        return my_op, dma_op, weight_byte_arr, bias_byte_arr, 





    def def_mul_decay_Vmem():
    
        IFM2_IS_FIRST_OPERAND = False


        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale = V_MEM_SCALE,
            zero_point = V_MEM_ZERO_POINT,
            name="v_mem"
        )

        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            #layout=NpuLayout.NHCWB16,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=DECAY_ADDR,
            scale = DECAY_SCALE,
            zero_point = DECAY_ZERO_POINT,
            name="decay"
        )



        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            #layout=NpuLayout.NHCWB16,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=DECAYED_MEM_ADDR,
            scale = DECAYED_MEM_SCALE,
            zero_point = DECAYED_MEM_ZERO_POINT,
            name="decayed_mem"
        )

        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )




        mul_decay_op = NpuElementWiseOperation(NpuElementWiseOp.MUL)
    
        #elementwise operation
        mul_decay_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        mul_decay_op.rescale = None

        #NpuBlockOperation
        mul_decay_op.ifm = ifm
        mul_decay_op.ifm2 = ifm2
        mul_decay_op.ifm2_scalar = None   #set if ifm2 is a scalar
        mul_decay_op.ofm = ofm
        mul_decay_op.kernel = None
        mul_decay_op.weights = []
        mul_decay_op.biases = []
        mul_decay_op.padding = None
        mul_decay_op.activation = activation

        block_config = get_block_config(mul_decay_op, ACCELERATOR)
        mul_decay_op.block_config = block_config
        mul_decay_op.rounding_mode = NpuRoundingMode.TFL
        mul_decay_op.fused_quantize = False
        mul_decay_op.ifm_upscale = NpuResamplingMode.NONE
        mul_decay_op.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, mul_decay_op, ACCELERATOR)



        return mul_decay_op




    def def_add_decayed_mem_in_curr():
        IFM2_IS_FIRST_OPERAND = False

        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            #layout=NpuLayout.NHCWB16,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=DECAYED_MEM_ADDR,
            scale = DECAYED_MEM_SCALE,
            zero_point = DECAYED_MEM_ZERO_POINT,
            name="decayed_mem"
        )

        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            #layout=NpuLayout.NHCWB16,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=IN_CURR_ADDR,
            scale = IN_CURR_SCALE,
            zero_point = IN_CURR_ZERO_POINT,
            name="in_curr"
        )




        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            #layout=NpuLayout.NHCWB16,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale=V_MEM_SCALE,
            zero_point=V_MEM_ZERO_POINT,
            name="updated_mem"
        )

        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )




        add_decayed_mem_in_curr = NpuElementWiseOperation(NpuElementWiseOp.ADD)

        #elementwise operation
        add_decayed_mem_in_curr.reversed_operands = IFM2_IS_FIRST_OPERAND
        add_decayed_mem_in_curr.rescale = None

        #NpuBlockOperation
        add_decayed_mem_in_curr.ifm = ifm
        add_decayed_mem_in_curr.ifm2 = ifm2
        add_decayed_mem_in_curr.ifm2_scalar = None   #set if ifm2 is a scalar
        add_decayed_mem_in_curr.ofm = ofm
        add_decayed_mem_in_curr.kernel = None
        add_decayed_mem_in_curr.weights = []
        add_decayed_mem_in_curr.biases = []
        add_decayed_mem_in_curr.padding = None
        add_decayed_mem_in_curr.activation = activation

        block_config = get_block_config(add_decayed_mem_in_curr, ACCELERATOR)
        add_decayed_mem_in_curr.block_config = block_config
        add_decayed_mem_in_curr.rounding_mode = NpuRoundingMode.TFL
        add_decayed_mem_in_curr.fused_quantize = False
        add_decayed_mem_in_curr.ifm_upscale = NpuResamplingMode.NONE
        add_decayed_mem_in_curr.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, add_decayed_mem_in_curr, ACCELERATOR)



        return add_decayed_mem_in_curr





    def def_check_spk_sub_v_mem_updated_vth():
        IFM2_IS_FIRST_OPERAND = False 

        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale = V_MEM_SCALE,
            zero_point = V_MEM_ZERO_POINT,
            name="v_mem_updated"
        )

        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=PARAMS_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=VTH_ADDR,
            scale = VTH_SCALE,
            zero_point = VTH_ZERO_POINT,
            name="vth"
        )




        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=OUTPUT_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=OUT_SPK_ADDR,
            scale=OUT_SPK_SCALE,
            zero_point=OUT_SPK_ZERO_POINT,
            name="out_spk"
        )

        #block_config = NpuShape3D(2, 2, 32)


        check_spk_lut_index = CHECK_SPK_LUT_INDEX
        activation = create_activation(
            activation_op=NpuActivationOp.TABLE_LOOKUP,
            min_val=None,
            max_val=None,
            lookup_table_index=check_spk_lut_index
        )

        #activation = create_activation(
            #activation_op=NpuActivationOp.NONE_OR_RELU,
            #min_val=None,
            #max_val=None
        #)


        # Define function that lut will approximate
        def check_positive(x_real):
            if x_real > 0:
                y_real = 1
            else:
                y_real = 0
        
            return y_real

        # It might be problematic to have the same scaling before and after LUT, currently is working though
        # if scale = 1, zero_point = 0, and dif = (v_mem - vth), where dif < 1, then it will only spike if dif is rounded to 1 (and not 0)
        check_spk_lut_dma_op, check_spk_lut_values = create_lut_and_dma(approximated_func=check_positive, lut_index=check_spk_lut_index, lut_region=LUT_REGION, data_type=ofm.data_type, 
                        scale_pre_lut=OUT_SPK_SCALE, zero_point_pre_lut=OUT_SPK_ZERO_POINT,
                        scale_post_lut=OUT_SPK_SCALE, zero_point_post_lut=OUT_SPK_ZERO_POINT,
                        accelerator=ACCELERATOR,
                        debug_mode=DEBUG_MODE
        )




        check_spk_sub_v_mem_updated_vth_op = NpuElementWiseOperation(NpuElementWiseOp.SUB)

        #elementwise operation
        check_spk_sub_v_mem_updated_vth_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        check_spk_sub_v_mem_updated_vth_op.rescale = None

        #NpuBlockOperation
        check_spk_sub_v_mem_updated_vth_op.ifm = ifm
        check_spk_sub_v_mem_updated_vth_op.ifm2 = ifm2
        check_spk_sub_v_mem_updated_vth_op.ifm2_scalar = None   #set if ifm2 is a scalar
        check_spk_sub_v_mem_updated_vth_op.ofm = ofm
        check_spk_sub_v_mem_updated_vth_op.kernel = None
        check_spk_sub_v_mem_updated_vth_op.weights = []
        check_spk_sub_v_mem_updated_vth_op.biases = []
        check_spk_sub_v_mem_updated_vth_op.padding = None
        check_spk_sub_v_mem_updated_vth_op.activation = activation
    
        block_config = get_block_config(check_spk_sub_v_mem_updated_vth_op, ACCELERATOR)
        check_spk_sub_v_mem_updated_vth_op.block_config = block_config
        check_spk_sub_v_mem_updated_vth_op.rounding_mode = NpuRoundingMode.TFL
        check_spk_sub_v_mem_updated_vth_op.fused_quantize = False
        check_spk_sub_v_mem_updated_vth_op.ifm_upscale = NpuResamplingMode.NONE
        check_spk_sub_v_mem_updated_vth_op.accumulator_type = NpuAccumulatorType.Int32


        #check_block_config_legal(block_config, check_spk_sub_v_mem_updated_vth_op, ACCELERATOR)



        return check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth_op, check_spk_lut_values, check_spk_lut_index



    def def_mul_vth_out_spk():

        IFM2_IS_FIRST_OPERAND = False


        #ifm = create_feature_map(
            #height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            #region=PARAMS_REGION,
            #layout=NpuLayout.NHWC,
            #data_type=NpuDataType.INT8,
            #fm_elem_size=1,
            #fm_addr=VTH_ADDR,
            #scale = VTH_SCALE,
            #zero_point = VTH_ZERO_POINT,
            #name="vth"
        #)
        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale = V_MEM_SCALE,
            zero_point = V_MEM_ZERO_POINT,
            name="vth"
        )

        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=OUTPUT_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=OUT_SPK_ADDR,
            scale = OUT_SPK_SCALE,
            zero_point = OUT_SPK_ZERO_POINT,
            name="out_spk"
        )


        # Same scaling as VTH (since reset is either 0 or 1)
        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            #layout=NpuLayout.NHCWB16,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=RESET_ADDR,
            scale = RESET_SCALE,
            zero_point = RESET_ZERO_POINT,
            name="reset"
        )


        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )




        mul_vth_out_spk_op = NpuElementWiseOperation(NpuElementWiseOp.MUL)
    
        #elementwise operation
        mul_vth_out_spk_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        mul_vth_out_spk_op.rescale = None

        #NpuBlockOperation
        mul_vth_out_spk_op.ifm = ifm
        mul_vth_out_spk_op.ifm2 = ifm2
        mul_vth_out_spk_op.ifm2_scalar = None   #set if ifm2 is a scalar
        mul_vth_out_spk_op.ofm = ofm
        mul_vth_out_spk_op.kernel = None
        mul_vth_out_spk_op.weights = []
        mul_vth_out_spk_op.biases = []
        mul_vth_out_spk_op.padding = None
        mul_vth_out_spk_op.activation = activation


        block_config = get_block_config(mul_vth_out_spk_op, ACCELERATOR)
        mul_vth_out_spk_op.block_config = block_config

        mul_vth_out_spk_op.rounding_mode = NpuRoundingMode.TFL
        mul_vth_out_spk_op.fused_quantize = False
        mul_vth_out_spk_op.ifm_upscale = NpuResamplingMode.NONE
        mul_vth_out_spk_op.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, mul_vth_out_spk_op, ACCELERATOR)



        return mul_vth_out_spk_op

    

    def def_sub_mem_updated_reset():

        IFM2_IS_FIRST_OPERAND = False


        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale = V_MEM_SCALE,
            zero_point = V_MEM_ZERO_POINT,
            name="v_mem"
        )


        # Same scaling as VTH (since reset is either 0 or 1)
        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            #layout=NpuLayout.NHCWB16,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=RESET_ADDR,
            scale = RESET_SCALE,
            zero_point = RESET_ZERO_POINT,
            name="reset"
        )


        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale = V_MEM_SCALE,
            zero_point = V_MEM_ZERO_POINT,
            name="v_mem_post_reset"
        )

        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )




        sub_v_mem_reset_op = NpuElementWiseOperation(NpuElementWiseOp.SUB)
    
        #elementwise operation
        sub_v_mem_reset_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        sub_v_mem_reset_op.rescale = None

        #NpuBlockOperation
        sub_v_mem_reset_op.ifm = ifm
        sub_v_mem_reset_op.ifm2 = ifm2
        sub_v_mem_reset_op.ifm2_scalar = None   #set if ifm2 is a scalar
        sub_v_mem_reset_op.ofm = ofm
        sub_v_mem_reset_op.kernel = None
        sub_v_mem_reset_op.weights = []
        sub_v_mem_reset_op.biases = []
        sub_v_mem_reset_op.padding = None
        sub_v_mem_reset_op.activation = activation

        block_config = get_block_config(sub_v_mem_reset_op, ACCELERATOR)
        sub_v_mem_reset_op.block_config = block_config
        sub_v_mem_reset_op.rounding_mode = NpuRoundingMode.TFL
        sub_v_mem_reset_op.fused_quantize = False
        sub_v_mem_reset_op.ifm_upscale = NpuResamplingMode.NONE
        sub_v_mem_reset_op.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, sub_v_mem_reset_op, ACCELERATOR)



        return sub_v_mem_reset_op
    


    def def_update_nxt_layer_reduce_sum_out_spk():


        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=OUTPUT_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8, fm_elem_size=1,
            fm_addr=OUT_SPK_ADDR,
            scale=OUT_SPK_SCALE, zero_point=OUT_SPK_ZERO_POINT,
            name="out_spk"
        )

        ofm = create_feature_map(
            height=1, width=1, depth=1,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8, fm_elem_size=1,
            fm_addr=UPDATE_NXT_LAYER_ADDR,
            scale=UPDATE_NXT_LAYER_SCALE,
            zero_point=UPDATE_NXT_LAYER_ZERO_POINT,
            name="update_nxt_layer"
        )


        kernel = NpuKernel(
            w=1, h=1, stride_x=1, stride_y=1,
            dilation_x=1, dilation_y=1
        )

        padding = NpuPadding(top=0, left=0, bottom=0, right=0)
   
        #block_config = NpuShape3D(2, 2, 8)


        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None
        )

        update_nxt_layer_reduce_sum_op = NpuPoolingOperation(NpuPoolingOp.REDUCE_SUM)
    
        #Pooling operation
        update_nxt_layer_reduce_sum_op.rescale = None

        #NpuBlockOperation
        update_nxt_layer_reduce_sum_op.ifm = ifm
        update_nxt_layer_reduce_sum_op.ifm2 = None
        update_nxt_layer_reduce_sum_op.ifm2_scalar = None   #set if ifm2 is a scalar
        update_nxt_layer_reduce_sum_op.ofm = ofm
        update_nxt_layer_reduce_sum_op.kernel = kernel
        update_nxt_layer_reduce_sum_op.weights = []
        update_nxt_layer_reduce_sum_op.biases = []
        update_nxt_layer_reduce_sum_op.padding = padding
        update_nxt_layer_reduce_sum_op.activation = activation

        block_config = get_block_config(update_nxt_layer_reduce_sum_op, ACCELERATOR)
        update_nxt_layer_reduce_sum_op.block_config = block_config
        update_nxt_layer_reduce_sum_op.rounding_mode = NpuRoundingMode.TFL
        update_nxt_layer_reduce_sum_op.fused_quantize = False
        update_nxt_layer_reduce_sum_op.ifm_upscale = NpuResamplingMode.NONE
        update_nxt_layer_reduce_sum_op.accumulator_type = NpuAccumulatorType.Default

        #check_block_config_legal(block_config, update_nxt_layer_reduce_sum_op, ACCELERATOR)

        return update_nxt_layer_reduce_sum_op




    def def_reset_time():

        IFM2_IS_FIRST_OPERAND = False


        ifm = create_feature_map(
            height=1, width=1, depth=1,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=TIME_NOT_UPDATED_ADDR,
            scale = TIME_NOT_UPDATED_SCALE,
            zero_point = TIME_NOT_UPDATED_ZERO_POINT,
            name="time_not_updated"
        )

        ifm2 = NpuFeatureMap()
        ifm2.quantization = NpuQuantization(1, 0)
        ifm2_scalar = 0

        ofm = create_feature_map(
            height=1, width=1, depth=1,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=TIME_NOT_UPDATED_ADDR,
            scale = TIME_NOT_UPDATED_SCALE,
            zero_point = TIME_NOT_UPDATED_ZERO_POINT,
            name="time_not_updated"
        )

        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=0,
            max_val=0,
        )




        reset_time_op = NpuElementWiseOperation(NpuElementWiseOp.MUL)
    
        #elementwise operation
        reset_time_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        reset_time_op.rescale = None

        #NpuBlockOperation
        reset_time_op.ifm = ifm
        reset_time_op.ifm2 = ifm2
        reset_time_op.ifm2_scalar = ifm2_scalar   #set if ifm2 is a scalar
        reset_time_op.ofm = ofm
        reset_time_op.kernel = None
        reset_time_op.weights = []
        reset_time_op.biases = []
        reset_time_op.padding = None
        reset_time_op.activation = activation
        print("reset time")
        block_config = get_block_config(reset_time_op, ACCELERATOR)
        reset_time_op.block_config = block_config
        reset_time_op.rounding_mode = NpuRoundingMode.TFL
        reset_time_op.fused_quantize = True 
        reset_time_op.ifm_upscale = NpuResamplingMode.NONE
        reset_time_op.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, reset_time_op, ACCELERATOR)



        return reset_time_op






    def layer0_merge_and_write(cms_name, header_out_filepath):

        # Define the individual NPU Operations
        dma_lut_op, exp_mul_lnb_time_op, decay_lut_values, decay_lut_index = def_decay_lut()
        fully_connected_op, dma_op, weight_byte_arr, bias_byte_arr = def_fullyconnected(IN_SPK_ADDR, IN_CURR_ADDR)
        mul_decay_op = def_mul_decay_Vmem()
        add_decayed_mem_in_curr = def_add_decayed_mem_in_curr()
        check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, check_spk_lut_values, check_spk_lut_index = def_check_spk_sub_v_mem_updated_vth()
        reset_mul_vth_out_spk_op = def_mul_vth_out_spk()
        sub_v_mem_reset_op = def_sub_mem_updated_reset()
        update_nxt_layer_reduce_sum_out_spk = def_update_nxt_layer_reduce_sum_out_spk()
        reset_time_op = def_reset_time()


        


        '''All Ops'''
        #npu_op_list = [dma_lut_op, exp_mul_lnb_time_op, dma_op, fully_connected_op, mul_decay_op, add_decayed_mem_in_curr, check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, reset_mul_vth_out_spk_op, sub_v_mem_reset_op, update_nxt_layer_reduce_sum_out_spk, reset_time_op]

        '''No LUT, No FC'''
        #npu_op_list = [mul_decay_op, add_decayed_mem_in_curr, reset_mul_vth_out_spk_op, sub_v_mem_reset_op, update_nxt_layer_reduce_sum_out_spk, reset_time_op]



        '''No activation'''
        #npu_op_list = [exp_mul_lnb_time_op, dma_op, fully_connected_op, mul_decay_op, add_decayed_mem_in_curr, check_spk_sub_v_mem_updated_vth, reset_mul_vth_out_spk_op, sub_v_mem_reset_op, update_nxt_layer_reduce_sum_out_spk, reset_time_op]

        '''No Reduced Sum'''
        #npu_op_list = [dma_lut_op, exp_mul_lnb_time_op, dma_op, fully_connected_op, mul_decay_op, add_decayed_mem_in_curr, check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, reset_mul_vth_out_spk_op, sub_v_mem_reset_op, reset_time_op]

        '''Only Reset time'''
        #npu_op_list = [reset_time_op]


        '''Only Mem Update reset'''
        #npu_op_list = [sub_v_mem_reset_op]


        '''Only FC Matmul'''
        npu_op_list = [dma_op, fully_connected_op]

        '''Only mul_vth_out_spk and update_nxt_reduced_sum'''
        #npu_op_list = [reset_mul_vth_out_spk_op, update_nxt_layer_reduce_sum_out_spk]

        '''Only mul_decay_op and add_decayed_mem_in_curr and mul_vth_out_spk and reset_time'''
        #npu_op_list = [mul_decay_op, add_decayed_mem_in_curr, reset_mul_vth_out_spk_op, reset_time_op]

        '''Only_mul_vth_out_spk_and_reset_time'''
        #npu_op_list = [reset_mul_vth_out_spk_op]


        '''Only decay_lut'''
        #npu_op_list = [dma_lut_op, exp_mul_lnb_time_op]

        # Merge
        lut_arr_contents_str = merge_lut_values_to_str([(decay_lut_values, decay_lut_index), (check_spk_lut_values, check_spk_lut_index)])
        lif_params_arr_contents_str = merge_lif_params_to_str(LN_BETA_QUANT_LIST, VTH_QUANT_LIST)
        cms_bytearr, register_cms = gen_cms(npu_op_list, ACCELERATOR, DEBUG_MODE)


        # For testing remove all ops but convolution op
        #print("cms_bytearr:\n", cms_bytearr)
        #import re
        ## Match little-endian 0x0002 → bytes: b'\x02\x00'
        #from cms_interpreter import cmd0_dict, cmd1_dict
        #pattern = re.compile(b'\x02\x00\x00\x00')

        ## Find all matches
        #matches = list(pattern.finditer(cms_bytearr))

        ## Print results
        #for match in matches:
            #print(f"Match at offset {match.start()}: {match.group()}")

        ##take first match
        ##cms_bytearr = matches[0]
        #print("new cms_byte_arr\n", cms_bytearr)

        from cms_interpreter import strip_cmds_from_register_command_stream
        dict_of_cmds_to_strip = {
            #'NPU_SET_IFM_REGION',
            #'NPU_SET_IFM_BASE0',
            #'NPU_SET_IFM_BASE1',
            #'NPU_SET_IFM_BASE2',
            #'NPU_SET_IFM_BASE3',

            #'NPU_SET_OFM_REGION',
            #'NPU_SET_OFM_BASE0',
            #'NPU_SET_OFM_BASE1',
            #'NPU_SET_OFM_BASE2',
            #'NPU_SET_OFM_BASE3',

            #'NPU_SET_WEIGHT_REGION',
            #'NPU_SET_WEIGHT_BASE',
            #'NPU_SET_WEIGHT_LENGTH',
            #'NPU_SET_SCALE_REGION',
            #'NPU_SET_SCALE_BASE',
            #'NPU_SET_SCALE_LENGTH'

        }
        
        #print("About to enter strip cmds()")
        new_register_cms, new_cmd_table = strip_cmds_from_register_command_stream(register_cms, dict_of_cmds_to_strip)
        
        #print("stripped commands\n", new_cmd_table)
        new_cms_byte_arr = npu_create_driver_payload(new_register_cms, ACCELERATOR)

        # Use the new cms
        #cms_bytearr = new_cms_byte_arr
        #register_cms = new_register_cms


        # Generate Dicts for writing to C
        sizes_dict, addr_dict, quant_param_dict = generate_dict_for_writing_defines_to_C_files(cms_name=cms_name, weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)



        write_cms_to_files(header_out_filepath, cms_bytearr, register_cms, cms_name, sizes_dict, addr_dict, quant_param_dict, lif_params_arr_contents_str, lut_arr_contents_str, weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)
    

    layer0_merge_and_write(cms_name, header_out_filepath)