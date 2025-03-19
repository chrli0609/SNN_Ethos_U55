# components.cmake

# component ARM::CMSIS:CORE@5.6.0
add_library(ARM_CMSIS_CORE_5_6_0 INTERFACE)
target_include_directories(ARM_CMSIS_CORE_5_6_0 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/Core/Include
)
target_compile_definitions(ARM_CMSIS_CORE_5_6_0 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)

# component ARM::CMSIS:DSP&Source@1.10.0
add_library(ARM_CMSIS_DSP_Source_1_10_0 OBJECT
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/BasicMathFunctions/BasicMathFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/BasicMathFunctions/BasicMathFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/BayesFunctions/BayesFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/BayesFunctions/BayesFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/CommonTables/CommonTables.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/CommonTables/CommonTablesF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/ComplexMathFunctions/ComplexMathFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/ComplexMathFunctions/ComplexMathFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/ControllerFunctions/ControllerFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/DistanceFunctions/DistanceFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/DistanceFunctions/DistanceFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/FastMathFunctions/FastMathFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/FastMathFunctions/FastMathFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/FilteringFunctions/FilteringFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/FilteringFunctions/FilteringFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/MatrixFunctions/MatrixFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/MatrixFunctions/MatrixFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/QuaternionMathFunctions/QuaternionMathFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/SVMFunctions/SVMFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/SVMFunctions/SVMFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/StatisticsFunctions/StatisticsFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/StatisticsFunctions/StatisticsFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/SupportFunctions/SupportFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/SupportFunctions/SupportFunctionsF16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/TransformFunctions/TransformFunctions.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Source/TransformFunctions/TransformFunctionsF16.c"
)
target_include_directories(ARM_CMSIS_DSP_Source_1_10_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Include
  ${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/PrivateInclude
)
target_compile_definitions(ARM_CMSIS_DSP_Source_1_10_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(ARM_CMSIS_DSP_Source_1_10_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(ARM_CMSIS_DSP_Source_1_10_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component ARM::CMSIS:NN Lib@4.0.0
add_library(ARM_CMSIS_NN_Lib_4_0_0 OBJECT
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ActivationFunctions/arm_nn_activation_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ActivationFunctions/arm_relu6_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ActivationFunctions/arm_relu_q15.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ActivationFunctions/arm_relu_q7.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/BasicMathFunctions/arm_elementwise_add_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/BasicMathFunctions/arm_elementwise_add_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/BasicMathFunctions/arm_elementwise_mul_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/BasicMathFunctions/arm_elementwise_mul_s16_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/BasicMathFunctions/arm_elementwise_mul_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConcatenationFunctions/arm_concatenation_s8_w.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConcatenationFunctions/arm_concatenation_s8_x.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConcatenationFunctions/arm_concatenation_s8_y.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConcatenationFunctions/arm_concatenation_s8_z.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_convolve_fast_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_convolve_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_convolve_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_convolve_wrapper_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_depthwise_conv_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/FullyConnectedFunctions/arm_fully_connected_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/FullyConnectedFunctions/arm_fully_connected_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/LSTMFunctions/arm_lstm_unidirectional_s8_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s8_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_lstm_step_s8_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_lstm_update_cell_state_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_lstm_update_output_s8_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_mat_mul_kernel_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_nntables.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/PoolingFunctions/arm_avgpool_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/PoolingFunctions/arm_avgpool_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/PoolingFunctions/arm_max_pool_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/PoolingFunctions/arm_max_pool_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/ReshapeFunctions/arm_reshape_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/SVDFunctions/arm_svdf_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/SVDFunctions/arm_svdf_state_s16_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/SoftmaxFunctions/arm_softmax_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/SoftmaxFunctions/arm_softmax_s8.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/SoftmaxFunctions/arm_softmax_s8_s16.c"
  "${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Source/SoftmaxFunctions/arm_softmax_u8.c"
)
target_include_directories(ARM_CMSIS_NN_Lib_4_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Include
)
target_compile_definitions(ARM_CMSIS_NN_Lib_4_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(ARM_CMSIS_NN_Lib_4_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(ARM_CMSIS_NN_Lib_4_0_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component ARM::ML Eval Kit:Common:API@1.0.0
add_library(ARM_ML_Eval_Kit_Common_API_1_0_0 OBJECT
  "${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/common/source/Classifier.cc"
  "${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/common/source/ImageUtils.cc"
  "${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/common/source/Mfcc.cc"
  "${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/common/source/Model.cc"
  "${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/common/source/TensorFlowLiteMicro.cc"
)
target_include_directories(ARM_ML_Eval_Kit_Common_API_1_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/common/include
)
target_compile_definitions(ARM_ML_Eval_Kit_Common_API_1_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(ARM_ML_Eval_Kit_Common_API_1_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(ARM_ML_Eval_Kit_Common_API_1_0_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component ARM::ML Eval Kit:Common:Log@1.0.0
add_library(ARM_ML_Eval_Kit_Common_Log_1_0_0 INTERFACE)
target_include_directories(ARM_ML_Eval_Kit_Common_Log_1_0_0 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/log/include
)
target_compile_definitions(ARM_ML_Eval_Kit_Common_Log_1_0_0 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)

# component ARM::ML Eval Kit:Common:Math@1.0.0
add_library(ARM_ML_Eval_Kit_Common_Math_1_0_0 OBJECT
  "${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/math/PlatformMath.cc"
)
target_include_directories(ARM_ML_Eval_Kit_Common_Math_1_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/log/include
  ${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/math/include
)
target_compile_definitions(ARM_ML_Eval_Kit_Common_Math_1_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(ARM_ML_Eval_Kit_Common_Math_1_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(ARM_ML_Eval_Kit_Common_Math_1_0_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component ARM::ML Eval Kit:Vision:Object detection@1.0.0
add_library(ARM_ML_Eval_Kit_Vision_Object_detection_1_0_0 OBJECT
  "${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/use_case/object_detection/src/DetectorPostProcessing.cc"
  "${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/use_case/object_detection/src/DetectorPreProcessing.cc"
  "${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/use_case/object_detection/src/YoloFastestModel.cc"
)
target_include_directories(ARM_ML_Eval_Kit_Vision_Object_detection_1_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/use_case/object_detection/include
  ${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/log/include
  ${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/math/include
)
target_compile_definitions(ARM_ML_Eval_Kit_Vision_Object_detection_1_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(ARM_ML_Eval_Kit_Vision_Object_detection_1_0_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(ARM_ML_Eval_Kit_Vision_Object_detection_1_0_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::BSP:External peripherals:CAMERA Sensor ARX3A0@1.3.0
add_library(AlifSemiconductor_BSP_External_peripherals_CAMERA_Sensor_ARX3A0_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/components/Source/arx3A0_camera_sensor.c"
)
target_include_directories(AlifSemiconductor_BSP_External_peripherals_CAMERA_Sensor_ARX3A0_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(AlifSemiconductor_BSP_External_peripherals_CAMERA_Sensor_ARX3A0_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_BSP_External_peripherals_CAMERA_Sensor_ARX3A0_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_BSP_External_peripherals_CAMERA_Sensor_ARX3A0_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::BSP:External peripherals:ILI9806E LCD panel@1.3.0
add_library(AlifSemiconductor_BSP_External_peripherals_ILI9806E_LCD_panel_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/components/Source/ILI9806E_LCD_panel.c"
)
target_include_directories(AlifSemiconductor_BSP_External_peripherals_ILI9806E_LCD_panel_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(AlifSemiconductor_BSP_External_peripherals_ILI9806E_LCD_panel_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_BSP_External_peripherals_ILI9806E_LCD_panel_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_BSP_External_peripherals_ILI9806E_LCD_panel_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::BSP:External peripherals:OSPI Flash ISSI@1.3.0
add_library(AlifSemiconductor_BSP_External_peripherals_OSPI_Flash_ISSI_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/components/Source/IS25WX256.c"
)
target_include_directories(AlifSemiconductor_BSP_External_peripherals_OSPI_Flash_ISSI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/components/Include
)
target_compile_definitions(AlifSemiconductor_BSP_External_peripherals_OSPI_Flash_ISSI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_BSP_External_peripherals_OSPI_Flash_ISSI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_BSP_External_peripherals_OSPI_Flash_ISSI_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:CDC@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_CDC_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_CDC200.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/cdc.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_CDC_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_CDC_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_CDC_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_CDC_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:CPI@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_CPI_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Camera_Sensor_i2c.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_CPI.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/cpi.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_CPI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_CPI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_CPI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_CPI_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:DMA@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_DMA_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_DMA.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/dma_ctrl.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/dma_op.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_DMA_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${SOLUTION_ROOT}/device/alif-ensemble-custom/RTE/Device/AE722F80F55D5LS_M55_HP
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_DMA_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_DMA_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_DMA_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:GPIO@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_GPIO_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_GPIO_Private.h"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_GPIO.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_GPIO_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_GPIO_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_GPIO_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_GPIO_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:I2C@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_I2C_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_I2C.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/i2c.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_I2C_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_I2C_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_I2C_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_I2C_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:I3C@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_I3C_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_I3C.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/i3c.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_I3C_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_I3C_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_I3C_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_I3C_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:MIPI CSI2@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_MIPI_CSI2_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/DPHY_CSI2.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_MIPI_CSI2.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/csi.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_MIPI_CSI2_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_MIPI_CSI2_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_MIPI_CSI2_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_MIPI_CSI2_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:MIPI DSI CSI2 DPHY@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_CSI2_DPHY_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/DPHY_Common.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_CSI2_DPHY_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_CSI2_DPHY_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_CSI2_DPHY_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_CSI2_DPHY_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:MIPI DSI@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/DPHY_DSI.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_MIPI_DSI.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/dsi.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:OSPI@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_OSPI_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_OSPI.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/ospi.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_OSPI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_OSPI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_OSPI_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_OSPI_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:PINCONF@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_PINCONF_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/pinconf.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_PINCONF_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_PINCONF_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_PINCONF_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_PINCONF_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:SOC Peripherals:USART@1.3.0
add_library(AlifSemiconductor_Device_SOC_Peripherals_USART_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source/Driver_USART.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/source/uart.c"
)
target_include_directories(AlifSemiconductor_Device_SOC_Peripherals_USART_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
)
target_compile_definitions(AlifSemiconductor_Device_SOC_Peripherals_USART_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_SOC_Peripherals_USART_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_SOC_Peripherals_USART_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component AlifSemiconductor::Device:Startup&C Startup@1.3.0
add_library(AlifSemiconductor_Device_Startup_C_Startup_1_3_0 OBJECT
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Device/common/source/clk.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Device/common/source/mpu_M55.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Device/common/source/system_M55.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Device/common/source/system_utils.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Device/common/source/tcm_partition.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Device/common/source/tgu_M55.c"
  "${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Device/core/M55_HP/source/startup_M55_HP.c"
)
target_include_directories(AlifSemiconductor_Device_Startup_C_Startup_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${SOLUTION_ROOT}/device/alif-ensemble-custom/RTE/Device/AE722F80F55D5LS_M55_HP
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Device/common/include
)
target_compile_definitions(AlifSemiconductor_Device_Startup_C_Startup_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(AlifSemiconductor_Device_Startup_C_Startup_1_3_0 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(AlifSemiconductor_Device_Startup_C_Startup_1_3_0 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component Arm::Machine Learning:NPU Support:Ethos-U Driver&Generic U55@1.23.2
add_library(Arm_Machine_Learning_NPU_Support_Ethos-U_Driver_Generic_U55_1_23_2 OBJECT
  "${CMSIS_PACK_ROOT}/ARM/ethos-u-core-driver/1.23.2/ethos_u_core_driver/src/ethosu_device_u55_u65.c"
  "${CMSIS_PACK_ROOT}/ARM/ethos-u-core-driver/1.23.2/ethos_u_core_driver/src/ethosu_driver.c"
  "${CMSIS_PACK_ROOT}/ARM/ethos-u-core-driver/1.23.2/ethos_u_core_driver/src/ethosu_pmu.c"
)
target_include_directories(Arm_Machine_Learning_NPU_Support_Ethos-U_Driver_Generic_U55_1_23_2 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/ARM/ethos-u-core-driver/1.23.2/ethos_u_core_driver/include
  ${CMSIS_PACK_ROOT}/ARM/ethos-u-core-driver/1.23.2/ethos_u_core_driver/src
)
target_compile_definitions(Arm_Machine_Learning_NPU_Support_Ethos-U_Driver_Generic_U55_1_23_2 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(Arm_Machine_Learning_NPU_Support_Ethos-U_Driver_Generic_U55_1_23_2 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(Arm_Machine_Learning_NPU_Support_Ethos-U_Driver_Generic_U55_1_23_2 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component tensorflow::Data Exchange:Serialization:flatbuffers&tensorflow@1.22.8
add_library(tensorflow_Data_Exchange_Serialization_flatbuffers_tensorflow_1_22_8 INTERFACE)
target_include_directories(tensorflow_Data_Exchange_Serialization_flatbuffers_tensorflow_1_22_8 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/tensorflow/flatbuffers/1.22.8/src/include
)
target_compile_definitions(tensorflow_Data_Exchange_Serialization_flatbuffers_tensorflow_1_22_8 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)

# component tensorflow::Data Processing:Math:gemmlowp fixed-point&tensorflow@1.22.8
add_library(tensorflow_Data_Processing_Math_gemmlowp_fixed-point_tensorflow_1_22_8 INTERFACE)
target_include_directories(tensorflow_Data_Processing_Math_gemmlowp_fixed-point_tensorflow_1_22_8 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/tensorflow/gemmlowp/1.22.8/src
)
target_compile_definitions(tensorflow_Data_Processing_Math_gemmlowp_fixed-point_tensorflow_1_22_8 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)

# component tensorflow::Data Processing:Math:kissfft&tensorflow@1.22.8
add_library(tensorflow_Data_Processing_Math_kissfft_tensorflow_1_22_8 OBJECT
  "${CMSIS_PACK_ROOT}/tensorflow/kissfft/1.22.8/src/kiss_fft.c"
  "${CMSIS_PACK_ROOT}/tensorflow/kissfft/1.22.8/src/tools/kiss_fftr.c"
)
target_include_directories(tensorflow_Data_Processing_Math_kissfft_tensorflow_1_22_8 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/tensorflow/kissfft/1.22.8/src
  ${CMSIS_PACK_ROOT}/tensorflow/kissfft/1.22.8/src/tools
)
target_compile_definitions(tensorflow_Data_Processing_Math_kissfft_tensorflow_1_22_8 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(tensorflow_Data_Processing_Math_kissfft_tensorflow_1_22_8 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(tensorflow_Data_Processing_Math_kissfft_tensorflow_1_22_8 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component tensorflow::Data Processing:Math:ruy&tensorflow@1.22.8
add_library(tensorflow_Data_Processing_Math_ruy_tensorflow_1_22_8 INTERFACE)
target_include_directories(tensorflow_Data_Processing_Math_ruy_tensorflow_1_22_8 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/tensorflow/ruy/1.22.8/src
)
target_compile_definitions(tensorflow_Data_Processing_Math_ruy_tensorflow_1_22_8 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)

# component tensorflow::Machine Learning:TensorFlow:Kernel Utils@1.22.8
add_library(tensorflow_Machine_Learning_TensorFlow_Kernel_Utils_1_22_8 OBJECT
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/kernels/kernel_util.cpp"
  "${SOLUTION_ROOT}/common/RTE/Machine_Learning/debug_log.cpp"
  "${SOLUTION_ROOT}/common/RTE/Machine_Learning/micro_time.cpp"
  "${SOLUTION_ROOT}/common/RTE/Machine_Learning/system_setup.cpp"
)
target_include_directories(tensorflow_Machine_Learning_TensorFlow_Kernel_Utils_1_22_8 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(tensorflow_Machine_Learning_TensorFlow_Kernel_Utils_1_22_8 PUBLIC
  $<$<COMPILE_LANGUAGE:C,CXX>:
    CMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE=CMSIS_device_header
  >
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(tensorflow_Machine_Learning_TensorFlow_Kernel_Utils_1_22_8 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(tensorflow_Machine_Learning_TensorFlow_Kernel_Utils_1_22_8 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component tensorflow::Machine Learning:TensorFlow:Kernel&Ethos-U@1.22.8
add_library(tensorflow_Machine_Learning_TensorFlow_Kernel_Ethos-U_1_22_8 OBJECT
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/c/common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/core/api/error_reporter.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/core/api/flatbuffer_conversions.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/core/api/op_resolver.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/core/api/tensor_utils.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/kernels/internal/quantization_util.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/all_ops_resolver.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/fake_micro_context.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/flatbuffer_utils.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/activations.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/activations_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/add_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/add_n.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/arg_min_max.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/assign_variable.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/batch_to_space_nd.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/broadcast_args.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/broadcast_to.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/call_once.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/cast.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/ceil.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/circular_buffer.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/circular_buffer_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/cmsis_nn/add.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/cmsis_nn/conv.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/cmsis_nn/depthwise_conv.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/cmsis_nn/mul.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/cmsis_nn/pooling.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/cmsis_nn/softmax.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/cmsis_nn/svdf.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/comparisons.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/concatenation.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/conv_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/cumsum.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/depth_to_space.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/depthwise_conv_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/dequantize.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/dequantize_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/detection_postprocess.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/div.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/elementwise.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/elu.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/ethos_u/ethosu.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/exp.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/expand_dims.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/fill.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/floor.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/floor_div.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/floor_mod.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/fully_connected_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/gather.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/gather_nd.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/hard_swish.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/hard_swish_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/if.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/kernel_runner.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/kernel_util.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/l2_pool_2d.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/l2norm.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/leaky_relu.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/leaky_relu_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/log_softmax.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/logical.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/logical_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/logistic.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/logistic_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/lstm_eval.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/maximum_minimum.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/micro_tensor_utils.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/mirror_pad.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/mul_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/neg.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/pack.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/pad.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/pooling_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/prelu.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/prelu_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/quantize.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/quantize_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/read_variable.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/reduce.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/reduce_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/reshape.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/resize_bilinear.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/round.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/select.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/shape.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/slice.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/softmax_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/space_to_batch_nd.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/space_to_depth.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/split.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/split_v.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/squared_difference.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/squeeze.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/strided_slice.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/sub.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/sub_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/svdf_common.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/tanh.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/transpose.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/transpose_conv.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/unpack.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/var_handle.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/while.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels/zeros_like.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/memory_helpers.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/memory_planner/linear_memory_planner.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/micro_allocation_info.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/micro_allocator.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/micro_context.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/micro_error_reporter.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/micro_graph.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/micro_interpreter.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/micro_profiler.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/micro_resource_variable.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/micro_string.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/micro_utils.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/mock_micro_graph.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/recording_micro_allocator.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/test_helper_custom_ops.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/test_helpers.cpp"
  "${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/schema/schema_utils.cpp"
)
target_include_directories(tensorflow_Machine_Learning_TensorFlow_Kernel_Ethos-U_1_22_8 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/c
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/core/api
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/kernels
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/kernels/internal
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/kernels/internal/optimized
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/kernels/internal/reference
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/kernels/internal/reference/integer_ops
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/arena_allocator
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/cortex_m_generic
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/kernels
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/memory_planner
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/schema
)
target_compile_definitions(tensorflow_Machine_Learning_TensorFlow_Kernel_Ethos-U_1_22_8 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(tensorflow_Machine_Learning_TensorFlow_Kernel_Ethos-U_1_22_8 PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(tensorflow_Machine_Learning_TensorFlow_Kernel_Ethos-U_1_22_8 PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# component tensorflow::Machine Learning:TensorFlow:Testing@1.22.8
add_library(tensorflow_Machine_Learning_TensorFlow_Testing_1_22_8 INTERFACE)
target_include_directories(tensorflow_Machine_Learning_TensorFlow_Testing_1_22_8 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/testing
)
target_compile_definitions(tensorflow_Machine_Learning_TensorFlow_Testing_1_22_8 INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
