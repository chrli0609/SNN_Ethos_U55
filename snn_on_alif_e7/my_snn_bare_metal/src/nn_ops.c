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


void print_input_output_tensor(struct ethosu_driver *drv) {


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
    }


uint32_t measure_layer0_start;
uint64_t start_cycles;
uint32_t start_mac;
uint32_t start_dpu;
uint32_t start_axi0_read;
uint32_t start_blockdep_stalls;

//This gets called by ethosu_inference_begin() in ethosu_cpu_cache.c
void ethosu_start_pmu_measure(struct ethosu_driver *drv, void *user_arg)
{
    if (MEASURE_MODE) {
        // Start measuring time
        measure_layer0_start = debug_start_timer();

        // 1. Enable PMU
        ETHOSU_PMU_Enable(drv);

        // 2. Configure counters
        ETHOSU_PMU_Set_EVTYPER(drv, 0, ETHOSU_PMU_MAC_ACTIVE);
        ETHOSU_PMU_Set_EVTYPER(drv, 1, ETHOSU_PMU_MAC_DPU_ACTIVE);
        ETHOSU_PMU_Set_EVTYPER(drv, 2, ETHOSU_PMU_AXI0_RD_TRANS_ACCEPTED);
        ETHOSU_PMU_Set_EVTYPER(drv, 3, ETHOSU_PMU_CC_STALLED_ON_BLOCKDEP);


        // 3. Reset counters
        ETHOSU_PMU_CYCCNT_Reset(drv);
        ETHOSU_PMU_EVCNTR_ALL_Reset(drv);

        // 4. Enable counters
        uint32_t mask = ETHOSU_PMU_CCNT_Msk | ETHOSU_PMU_CNT1_Msk | ETHOSU_PMU_CNT2_Msk | ETHOSU_PMU_CNT3_Msk | ETHOSU_PMU_CNT4_Msk;
        ETHOSU_PMU_CNTR_Enable(drv, mask);
        start_cycles = ETHOSU_PMU_Get_CCNTR(drv);
        start_mac   = ETHOSU_PMU_Get_EVCNTR(drv, 0);
        start_dpu   = ETHOSU_PMU_Get_EVCNTR(drv, 1);
        start_axi0_read   = ETHOSU_PMU_Get_EVCNTR(drv, 2);
        start_blockdep_stalls   = ETHOSU_PMU_Get_EVCNTR(drv, 3);

    }

}

void ethosu_inference_end(struct ethosu_driver *drv, void *user_arg)
{
    // Your custom end‑of‑inference hook
    // e.g., log results or stop the timer
    if (MEASURE_MODE) {
        uint32_t measure_layer0_elapsed_ticks = debug_end_timer(measure_layer0_start);


        uint64_t end_cycles = ETHOSU_PMU_Get_CCNTR(drv);
        uint32_t end_mac    = ETHOSU_PMU_Get_EVCNTR(drv, 0);
        uint32_t end_dpu    = ETHOSU_PMU_Get_EVCNTR(drv, 1);
        uint32_t end_axi0_read    = ETHOSU_PMU_Get_EVCNTR(drv, 2);
        uint32_t end_blockdep_stalls    = ETHOSU_PMU_Get_EVCNTR(drv, 3);



        // 7. Disable counters
        ETHOSU_PMU_CNTR_Disable(drv, ETHOSU_PMU_MAC_ACTIVE);
        ETHOSU_PMU_CNTR_Disable(drv, ETHOSU_PMU_MAC_ACTIVE_32BIT);
        ETHOSU_PMU_CNTR_Disable(drv, ETHOSU_PMU_AXI0_RD_TRANS_ACCEPTED);
        ETHOSU_PMU_CNTR_Disable(drv, ETHOSU_PMU_CC_STALLED_ON_BLOCKDEP);

        // 8. Read results
        //uint32_t mac_active = ETHOSU_PMU_Get_EVCNTR(drv, 0);
        //uint32_t idle_cycles = ETHOSU_PMU_Get_EVCNTR(drv, 1);
        //uint32_t block_stalls = ETHOSU_PMU_Get_EVCNTR(drv, 2);
        //uint32_t axi_stalls = ETHOSU_PMU_Get_EVCNTR(drv, 3);

        // 10. Disable PMU if no longer needed
        ETHOSU_PMU_Disable(drv);

        uint64_t cycles = end_cycles - start_cycles;
        uint32_t macs   = end_mac   - start_mac;
        uint32_t cc_dpus_active   = end_dpu   - start_dpu;
        uint32_t axi0_reads   = end_axi0_read   - start_axi0_read;
        uint32_t blockdep_stalls   = end_blockdep_stalls   - start_blockdep_stalls;


        // 9. Report or log
        //double utilization = (double)mac_active / (double)(mac_active + idle_cycles + block_stalls + axi_stalls);
        //printf("NPU_active + NPU_IDLE should be equal to NPU Cycles:%" PRIu32 "\n", mac_active+idle_cycles);
        printf("Npu cycles for it: %d = %" PRIu64 "\n", global_it, cycles);
        printf("Npu MAC Active for it: %d = %" PRIu32 "\n", global_it, macs);
        printf("Npu MAC 8bit Active for it: %d = %" PRIu32 "\n", global_it, cc_dpus_active);
        printf("Npu AXI0 Reads for it: %d = %" PRIu32 "\n", global_it, axi0_reads);
        printf("Npu Blockdep Stalls for it: %d = %" PRIu32 "\n", global_it, blockdep_stalls);
        //printf("Utilization: %f\n", utilization);
        //printf("\tidle_cycles: %" PRIu32 "\n", idle_cycles);
        //printf("\tblock_stalls: %" PRIu32 "\n", block_stalls);
        //printf("\taxi_stalls: %" PRIu32 "\n", axi_stalls);

        printf("Investigate start and end values:\n");
        printf("\tNPU Cycles:\t\t\tstart: %" PRIu64 "\tend: %" PRIu64"\n", start_cycles, end_cycles);
        printf("\tNPU MAC Operations:\t\tstart: %" PRIu32 "\tend: %" PRIu32"\n", start_mac, end_mac);
        printf("\tNPU MAC 8-bit Operations:\tstart: %" PRIu32 "\tend: %" PRIu32"\n", start_dpu, end_dpu);
        printf("\tNPU AXI0 Reads:\t\t\tstart: %" PRIu32 "\tend: %" PRIu32"\n", start_axi0_read, end_axi0_read);
        printf("\tNPU BlockDep Stalls:\t\tstart: %" PRIu32 "\tend: %" PRIu32"\n", start_blockdep_stalls, end_blockdep_stalls);


        // Print CPU timer results
        printf("Ticks elapsed for layer once in it: %d = %d \xC2\xB5s\n", global_it, measure_layer0_elapsed_ticks);
        //printf("Ticks elapsed for layer once in it: %d = %d ms\n", global_it, measure_layer0_elapsed_ticks);
        //printf("Ticks elapsed for layer once in it: %d = %" PRIu64 " \n", global_it, cycles);

    }
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








