cmake_minimum_required(VERSION 3.27)

# Roots
include("../roots.cmake")

set(CONTEXT simple_code_test.debug+Alif-E7-M55-HP)
set(TARGET ${CONTEXT})
set(OUT_DIR "${SOLUTION_ROOT}/out/simple_code_test/Alif-E7-M55-HP/debug")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BIN_FILE "simple_code_test.bin")
set(LD_SCRIPT "${SOLUTION_ROOT}/simple_code_test/linker/alif-e7-m55-hp.ld")
set(LD_SCRIPT_PP ${LD_SCRIPT})

# Processor Options
set(CPU Cortex-M55)
set(FPU DP_FPU)
set(DSP DSP)
set(MVE FP_FVE)
set(BYTE_ORDER Little-endian)

# Toolchain config map
include("toolchain.cmake")

# Setup project
project(${CONTEXT} LANGUAGES CXX C)

# Compilation database
add_custom_target(database
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" "${OUT_DIR}"
)
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES})

# Setup context
add_executable(${CONTEXT})
set_target_properties(${CONTEXT} PROPERTIES PREFIX "" SUFFIX ".elf" OUTPUT_NAME "simple_code_test")
set_target_properties(${CONTEXT} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})
add_library(${CONTEXT}_GLOBAL INTERFACE)

# Includes
target_include_directories(${CONTEXT} PUBLIC
  $<$<COMPILE_LANGUAGE:C,CXX>:
    ${SOLUTION_ROOT}/simple_code_test
    ${SOLUTION_ROOT}/simple_code_test/include
    ${SOLUTION_ROOT}/board
    ${SOLUTION_ROOT}/libs/board
    ${SOLUTION_ROOT}/libs/common_app_utils/logging
    ${SOLUTION_ROOT}/libs/common_app_utils/fault_handler
    ${SOLUTION_ROOT}/device/alif-ensemble-custom/Board
  >
  ${SOLUTION_ROOT}/simple_code_test/RTE/_debug_Alif-E7-M55-HP
  ${SOLUTION_ROOT}/device/alif-ensemble-custom/RTE/Device/AE722F80F55D5LS_M55_HP
  ${CMSIS_PACK_ROOT}/ARM/CMSIS-NN/4.0.0/Include
  ${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/Core/Include
  ${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/Include
  ${CMSIS_PACK_ROOT}/ARM/CMSIS/5.9.0/CMSIS/DSP/PrivateInclude
  ${CMSIS_PACK_ROOT}/ARM/ethos-u-core-driver/1.23.2/ethos_u_core_driver/include
  ${CMSIS_PACK_ROOT}/ARM/ethos-u-core-driver/1.23.2/ethos_u_core_driver/src
  ${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/application/api/common/include
  ${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/log/include
  ${CMSIS_PACK_ROOT}/ARM/ml-embedded-eval-kit-uc-api/22.8.0-Beta/source/math/include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Alif_CMSIS/Source
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Device/common/include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/Device/core/M55_HP/include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/components/Include
  ${CMSIS_PACK_ROOT}/AlifSemiconductor/Ensemble/1.3.0/drivers/include
  ${CMSIS_PACK_ROOT}/tensorflow/flatbuffers/1.22.8/src/include
  ${CMSIS_PACK_ROOT}/tensorflow/gemmlowp/1.22.8/src
  ${CMSIS_PACK_ROOT}/tensorflow/kissfft/1.22.8/src
  ${CMSIS_PACK_ROOT}/tensorflow/kissfft/1.22.8/src/tools
  ${CMSIS_PACK_ROOT}/tensorflow/ruy/1.22.8/src
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
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/micro/testing
  ${CMSIS_PACK_ROOT}/tensorflow/tensorflow-lite-micro/1.22.8/tensorflow/lite/schema
  ${SOLUTION_ROOT}/common/include
  ${SOLUTION_ROOT}/device/alif-ensemble-custom/include
  ${SOLUTION_ROOT}/device/alif-ensemble-custom/src
)

# Defines
target_compile_definitions(${CONTEXT} PUBLIC
  $<$<COMPILE_LANGUAGE:C,CXX>:
    ACTIVATION_BUF_SZ=532480
    ETHOSU55
    COPY_VECTORS
    NPU_QCONFIG=1
    NPU_REGIONCFG_0=1
    NPU_REGIONCFG_1=0
    NPU_REGIONCFG_2=1
    AXI_LIMIT0_MEM_TYPE=3
    AXI_LIMIT1_MEM_TYPE=3
    AXI_LIMIT2_MEM_TYPE=3
    AXI_LIMIT3_MEM_TYPE=3
    USART_DRV_NUM=2
    M55_HP
    _RTE_
  >
)

# Compile options
target_compile_options(${CONTEXT} PUBLIC
  $<$<COMPILE_LANGUAGE:C>:
    "SHELL:${CC_CPU}"
    "SHELL:${CC_FLAGS}"
    "SHELL:${CC_BYTE_ORDER}"
    "SHELL:-std=c99"
    "SHELL:-flax-vector-conversions"
    "SHELL:-DARM_MATH_AUTOVECTORIZE"
    "SHELL:-funsigned-char"
    "SHELL:-fdata-sections"
    "SHELL:-fno-function-sections"
    "SHELL:-include \"RTE_Components.h\""
  >
  $<$<COMPILE_LANGUAGE:C,CXX>:
    "SHELL:${_PI}\"${SOLUTION_ROOT}/simple_code_test/RTE/_debug_Alif-E7-M55-HP/Pre_Include_Global.h\""
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    "SHELL:${CXX_CPU}"
    "SHELL:${CXX_FLAGS}"
    "SHELL:${CXX_BYTE_ORDER}"
    "SHELL:-std=c++14"
    "SHELL:-fno-exceptions"
    "SHELL:-fno-rtti"
    "SHELL:-fno-unwind-tables"
    "SHELL:-funsigned-char"
    "SHELL:-fdata-sections"
    "SHELL:-fno-function-sections"
    "SHELL:-include \"RTE_Components.h\""
  >
)

# Compile Options Abstractions
add_library(${CONTEXT}_ABSTRACTIONS INTERFACE)
cbuild_set_options_flags(CXX "none" "on" "" "" CXX_OPTIONS_FLAGS_${CONTEXT})
cbuild_set_options_flags(CC "none" "on" "" "" CC_OPTIONS_FLAGS_${CONTEXT})
target_compile_options(${CONTEXT}_ABSTRACTIONS INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:
    "SHELL:${CXX_OPTIONS_FLAGS_${CONTEXT}}"
  >
  $<$<COMPILE_LANGUAGE:C>:
    "SHELL:${CC_OPTIONS_FLAGS_${CONTEXT}}"
  >
)

# Add groups and components
include("groups.cmake")
include("components.cmake")

target_link_libraries(${CONTEXT} PUBLIC
  Group_Main
  Group_Common
  Group_GPIO
  Group_Retarget
  Group_Init
  Group_Npu
  ARM_CMSIS_CORE_5_6_0
  ARM_CMSIS_DSP_Source_1_10_0
  ARM_CMSIS_NN_Lib_4_0_0
  ARM_ML_Eval_Kit_Common_API_1_0_0
  ARM_ML_Eval_Kit_Common_Log_1_0_0
  ARM_ML_Eval_Kit_Common_Math_1_0_0
  AlifSemiconductor_BSP_External_peripherals_CAMERA_Sensor_ARX3A0_1_3_0
  AlifSemiconductor_BSP_External_peripherals_ILI9806E_LCD_panel_1_3_0
  AlifSemiconductor_BSP_External_peripherals_OSPI_Flash_ISSI_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_CDC_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_CPI_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_DMA_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_GPIO_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_I2C_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_I3C_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_MIPI_CSI2_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_CSI2_DPHY_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_MIPI_DSI_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_OSPI_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_PINCONF_1_3_0
  AlifSemiconductor_Device_SOC_Peripherals_USART_1_3_0
  AlifSemiconductor_Device_Startup_C_Startup_1_3_0
  Arm_Machine_Learning_NPU_Support_Ethos-U_Driver_Generic_U55_1_23_2
  tensorflow_Data_Exchange_Serialization_flatbuffers_tensorflow_1_22_8
  tensorflow_Data_Processing_Math_gemmlowp_fixed-point_tensorflow_1_22_8
  tensorflow_Data_Processing_Math_kissfft_tensorflow_1_22_8
  tensorflow_Data_Processing_Math_ruy_tensorflow_1_22_8
  tensorflow_Machine_Learning_TensorFlow_Kernel_Utils_1_22_8
  tensorflow_Machine_Learning_TensorFlow_Kernel_Ethos-U_1_22_8
  tensorflow_Machine_Learning_TensorFlow_Testing_1_22_8
)

# Linker options
target_link_options(${CONTEXT} PUBLIC
  "SHELL:${LD_CPU}"
  "SHELL:${_LS}\"${LD_SCRIPT_PP}\""
  "SHELL:--specs=nosys.specs"
  "SHELL:--entry=Reset_Handler"
  "SHELL:-Wl,-Map=.map,--cref,-print-memory-usage,--gc-sections,--no-warn-rwx-segments"
)
set_target_properties(${CONTEXT} PROPERTIES LINK_DEPENDS ${LD_SCRIPT})

# Bin Conversion
 add_custom_command(TARGET ${CONTEXT} POST_BUILD COMMAND ${CMAKE_OBJCOPY} ${ELF2BIN})
