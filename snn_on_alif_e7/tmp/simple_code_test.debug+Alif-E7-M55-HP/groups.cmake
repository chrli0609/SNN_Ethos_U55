# groups.cmake

# group Main
add_library(Group_Main OBJECT
  "${SOLUTION_ROOT}/simple_code_test/src/main.c"
  "${SOLUTION_ROOT}/simple_code_test/src/nn_ops.c"
  "${SOLUTION_ROOT}/simple_code_test/src/extra_funcs.c"
  "${SOLUTION_ROOT}/simple_code_test/src/nn_data_structure.c"
  "${SOLUTION_ROOT}/simple_code_test/src/init_nn_model.c"
  "${SOLUTION_ROOT}/simple_code_test/nn_ops/fc_lif_layer_0.c"
  "${SOLUTION_ROOT}/simple_code_test/nn_ops/fc_lif_layer_1.c"
)
target_include_directories(Group_Main PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(Group_Main PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(Group_Main PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(Group_Main PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# group Common
add_library(Group_Common INTERFACE)
target_include_directories(Group_Common INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${SOLUTION_ROOT}/common/include
)
target_compile_definitions(Group_Common INTERFACE
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)

# group GPIO
add_library(Group_GPIO OBJECT
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/src/gpio_wrapper.c"
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/src/GpioSignal.cpp"
)
target_include_directories(Group_GPIO PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${SOLUTION_ROOT}/device/alif-ensemble-custom/include
)
target_compile_definitions(Group_GPIO PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(Group_GPIO PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(Group_GPIO PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# group Retarget
add_library(Group_Retarget OBJECT
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/src/retarget.c"
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/src/uart_stdout.c"
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/src/retarget_sbrk.c"
)
target_include_directories(Group_Retarget PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${SOLUTION_ROOT}/device/alif-ensemble-custom/src
)
target_compile_definitions(Group_Retarget PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(Group_Retarget PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(Group_Retarget PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# group Init
add_library(Group_Init OBJECT
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/src/BoardInit.c"
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/Board/devkit_gen2/board_init.c"
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/src/ospi_flash.c"
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/src/mpu_M55_region_config.c"
)
target_include_directories(Group_Init PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
  ${SOLUTION_ROOT}/device/alif-ensemble-custom/include
)
target_compile_definitions(Group_Init PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(Group_Init PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(Group_Init PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)

# group Npu
add_library(Group_Npu OBJECT
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/src/ethosu_cpu_cache.c"
  "${SOLUTION_ROOT}/device/alif-ensemble-custom/src/ethosu_platform_callbacks.c"
)
target_include_directories(Group_Npu PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(Group_Npu PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(Group_Npu PUBLIC
  $<TARGET_PROPERTY:${CONTEXT},INTERFACE_COMPILE_OPTIONS>
)
target_link_libraries(Group_Npu PUBLIC
  ${CONTEXT}_ABSTRACTIONS
)
