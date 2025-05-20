#include "include/nn_ops.h"






#include <stdio.h>
#include <stddef.h>




#include "ethosu_driver.h"
//#include "include/extra_funcs.h"







//mydebug
#include "include/extra_funcs.h"    //SysTick_Handler and start and end timer
#include "nn_data_structure.h"
extern int DEBUG_MODE;
extern int MEASURE_MODE;
extern int global_it;





#include <inttypes.h>   //for PRIu64
#include "pmu_ethosu.h" //For all ethosu PMU Functions


uint32_t measure_layer0_start;

//This gets called by ethosu_inference_begin() in ethosu_cpu_cache.c
void ethosu_start_pmu_measure(struct ethosu_driver *drv, void *user_arg)
{
    // Your custom start‑of‑inference hook
    // e.g., toggle an LED or start a timer
    if (MEASURE_MODE) {
        printf("Before NPU OP Start\n");
        int8_t* input = (int8_t *)(intptr_t)drv->job.base_addr[5];
        int8_t* output = (int8_t *)(intptr_t)drv->job.base_addr[6];
        size_t input_size = drv->job.base_addr_size[5];
        size_t output_size = drv->job.base_addr_size[6];
        // Print input
        printf("input:\n");
        for (size_t i = 0; i < input_size; i++){
            printf("%" PRIu8 ", ", input[i]);
        }
        printf("\n");

        // Print output
        printf("output:\n");
        for (size_t i = 0; i < output_size; i++){
            printf("%" PRIu8 ", ", output[i]);
        }
        printf("\n");
        measure_layer0_start = debug_start_timer();

        // 1. Enable PMU
        ETHOSU_PMU_Enable(drv);

        // 2. Configure counters
        ETHOSU_PMU_Set_EVTYPER(drv, 0, ETHOSU_PMU_NPU_IDLE);
        ETHOSU_PMU_Set_EVTYPER(drv, 1, ETHOSU_PMU_NPU_IDLE);
        ETHOSU_PMU_Set_EVTYPER(drv, 2, ETHOSU_PMU_CYCLE);
        ETHOSU_PMU_Set_EVTYPER(drv, 3, ETHOSU_PMU_AXI0_RD_TRAN_REQ_STALLED);


        // 3. Reset counters
        ETHOSU_PMU_CYCCNT_Reset(drv);
        ETHOSU_PMU_EVCNTR_ALL_Reset(drv);

        // 4. Enable counters
        uint32_t mask = ETHOSU_PMU_CCNT_Msk | ETHOSU_PMU_CNT1_Msk | ETHOSU_PMU_CNT2_Msk | ETHOSU_PMU_CNT3_Msk | ETHOSU_PMU_CNT4_Msk;
        ETHOSU_PMU_CNTR_Enable(drv, mask);

    }

}

void ethosu_inference_end(struct ethosu_driver *drv, void *user_arg)
{
    // Your custom end‑of‑inference hook
    // e.g., log results or stop the timer
    if (MEASURE_MODE) {
        uint32_t measure_layer0_elapsed_ticks = debug_end_timer(measure_layer0_start);

        // 7. Disable counters

        ETHOSU_PMU_CNTR_Disable(drv, ETHOSU_PMU_CYCLE);

        // 8. Read results
        uint64_t cycles = ETHOSU_PMU_Get_CCNTR(drv);
        uint32_t mac_active = ETHOSU_PMU_Get_EVCNTR(drv, 0);
        uint32_t idle_cycles = ETHOSU_PMU_Get_EVCNTR(drv, 1);
        uint32_t block_stalls = ETHOSU_PMU_Get_EVCNTR(drv, 2);
        uint32_t axi_stalls = ETHOSU_PMU_Get_EVCNTR(drv, 3);

        // 10. Disable PMU if no longer needed
        ETHOSU_PMU_Disable(drv);

        // 9. Report or log
        double utilization = (double)mac_active / (double)(mac_active + idle_cycles + block_stalls + axi_stalls);
        printf("NPU_active + NPU_IDLE should be equal to NPU Cycles:%" PRIu32 "\n", mac_active+idle_cycles);
        printf("NPU cycles: %" PRIu64 "\n", cycles);
        printf("MAC active: %" PRIu32 "\n", mac_active);
        printf("Utilization: %f\n", utilization);
        printf("\tidle_cycles: %" PRIu32 "\n", idle_cycles);
        printf("\tblock_stalls: %" PRIu32 "\n", block_stalls);
        printf("\taxi_stalls: %" PRIu32 "\n", axi_stalls);

        // Print CPU timer results
        //printf("Ticks elapsed for layer once in it: %d = %d\n", global_it, measure_layer0_elapsed_ticks);
        printf("Ticks elapsed for layer once in it: %d = %" PRIu64 " \xC2\xB5s\n", global_it, cycles);

    }
        int8_t* input = (int8_t *)(intptr_t)drv->job.base_addr[5];
        int8_t* output = (int8_t *)(intptr_t)drv->job.base_addr[6];
        size_t input_size = drv->job.base_addr_size[5];
        size_t output_size = drv->job.base_addr_size[6];

        printf("input_size: %d\n", input_size);
        printf("output_size: %d\n", output_size);
        printf("Atfter NPU OP\n");
        // Print input
        printf("input:\n");
        for (size_t i = 0; i < input_size; i++){
            printf("%" PRIu8 ", ", input[i]);
        }
        printf("\n");

        // Print output
        printf("output:\n");
        for (size_t i = 0; i < output_size; i++){
            printf("%" PRIu8 ", ", output[i]);
        }
        printf("\n");
}





//#include <cmsis_gcc.h>
//#include <pmu_armv8.h>

//#include "cmsis_compiler.h"    // or <cmsis_gcc.h> etc.
//#include "core_cm55.h"         // device-specific core header
//#include "pmu_armv8.h"         // now safe to include PMU API


int run_cms(
    const uint8_t* command_stream,
    size_t command_stream_size,
    uint64_t* base_addrs,
    size_t* base_addrs_size,
    int num_tensors

) {


    // Reserve the Ethos-U driver
    struct ethosu_driver* drv = ethosu_reserve_driver();
    if (!drv) {
        printf("Failed to reserve Ethos-U driver\n");
        return -1;
    }

    if (DEBUG_MODE) {
        printf("reserved ethosu_drv: %p\n", drv);
        printf("Before invoke_v3: &drv = %p, drv = %p\n", (void*)&drv, (void*)drv);
    }


    //uint32_t measure_layer0_start;

    if(ethosu_invoke_v3(drv, command_stream, command_stream_size, 
        base_addrs, base_addrs_size, num_tensors, NULL) != 0) {
        printf("Invoke_v3 Failed\n");
        return -1;
    }

    //mydebug
    if (DEBUG_MODE) { printf("After invoke_v3: &drv = %p, drv = %p\n", (void*)&drv, (void*)drv); }


    ethosu_release_driver(drv);

    if (DEBUG_MODE) { printf("Driver release successfully\n"); }



    return 0;

}








