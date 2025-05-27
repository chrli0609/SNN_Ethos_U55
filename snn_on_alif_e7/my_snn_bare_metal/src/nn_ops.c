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

const char* event_type_names[75] = {
    "ETHOSU_PMU_NO_EVENT",
    "ETHOSU_PMU_CYCLE",
    "ETHOSU_PMU_NPU_IDLE",
    "ETHOSU_PMU_CC_STALLED_ON_BLOCKDEP",
    "ETHOSU_PMU_CC_STALLED_ON_SHRAM_RECONFIG",
    "ETHOSU_PMU_NPU_ACTIVE",
    "ETHOSU_PMU_MAC_ACTIVE",
    "ETHOSU_PMU_MAC_ACTIVE_8BIT",
    "ETHOSU_PMU_MAC_ACTIVE_16BIT",
    "ETHOSU_PMU_MAC_DPU_ACTIVE",
    "ETHOSU_PMU_MAC_STALLED_BY_WD_ACC",
    "ETHOSU_PMU_MAC_STALLED_BY_WD",
    "ETHOSU_PMU_MAC_STALLED_BY_ACC",
    "ETHOSU_PMU_MAC_STALLED_BY_IB",
    "ETHOSU_PMU_MAC_ACTIVE_32BIT",
    "ETHOSU_PMU_MAC_STALLED_BY_INT_W",
    "ETHOSU_PMU_MAC_STALLED_BY_INT_ACC",
    "ETHOSU_PMU_AO_ACTIVE",
    "ETHOSU_PMU_AO_ACTIVE_8BIT",
    "ETHOSU_PMU_AO_ACTIVE_16BIT",
    "ETHOSU_PMU_AO_STALLED_BY_OFMP_OB",
    "ETHOSU_PMU_AO_STALLED_BY_OFMP",
    "ETHOSU_PMU_AO_STALLED_BY_OB",
    "ETHOSU_PMU_AO_STALLED_BY_ACC_IB",
    "ETHOSU_PMU_AO_STALLED_BY_ACC",
    "ETHOSU_PMU_AO_STALLED_BY_IB",
    "ETHOSU_PMU_WD_ACTIVE",
    "ETHOSU_PMU_WD_STALLED",
    "ETHOSU_PMU_WD_STALLED_BY_WS",
    "ETHOSU_PMU_WD_STALLED_BY_WD_BUF",
    "ETHOSU_PMU_WD_PARSE_ACTIVE",
    "ETHOSU_PMU_WD_PARSE_STALLED",
    "ETHOSU_PMU_WD_PARSE_STALLED_IN",
    "ETHOSU_PMU_WD_PARSE_STALLED_OUT",
    "ETHOSU_PMU_WD_TRANS_WS",
    "ETHOSU_PMU_WD_TRANS_WB",
    "ETHOSU_PMU_WD_TRANS_DW0",
    "ETHOSU_PMU_WD_TRANS_DW1",
    "ETHOSU_PMU_AXI0_RD_TRANS_ACCEPTED",
    "ETHOSU_PMU_AXI0_RD_TRANS_COMPLETED",
    "ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED",
    "ETHOSU_PMU_AXI0_RD_TRAN_REQ_STALLED",
    "ETHOSU_PMU_AXI0_WR_TRANS_ACCEPTED",
    "ETHOSU_PMU_AXI0_WR_TRANS_COMPLETED_M",
    "ETHOSU_PMU_AXI0_WR_TRANS_COMPLETED_S",
    "ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN",
    "ETHOSU_PMU_AXI0_WR_TRAN_REQ_STALLED",
    "ETHOSU_PMU_AXI0_WR_DATA_BEAT_STALLED",
    "ETHOSU_PMU_AXI0_ENABLED_CYCLES",
    "ETHOSU_PMU_AXI0_RD_STALL_LIMIT",
    "ETHOSU_PMU_AXI0_WR_STALL_LIMIT",
    "ETHOSU_PMU_AXI_LATENCY_ANY",
    "ETHOSU_PMU_AXI_LATENCY_32",
    "ETHOSU_PMU_AXI_LATENCY_64",
    "ETHOSU_PMU_AXI_LATENCY_128",
    "ETHOSU_PMU_AXI_LATENCY_256",
    "ETHOSU_PMU_AXI_LATENCY_512",
    "ETHOSU_PMU_AXI_LATENCY_1024",
    "ETHOSU_PMU_ECC_DMA",
    "ETHOSU_PMU_ECC_SB0",
    "ETHOSU_PMU_AXI1_RD_TRANS_ACCEPTED",
    "ETHOSU_PMU_AXI1_RD_TRANS_COMPLETED",
    "ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED",
    "ETHOSU_PMU_AXI1_RD_TRAN_REQ_STALLED",
    "ETHOSU_PMU_AXI1_WR_TRANS_ACCEPTED",
    "ETHOSU_PMU_AXI1_WR_TRANS_COMPLETED_M",
    "ETHOSU_PMU_AXI1_WR_TRANS_COMPLETED_S",
    "ETHOSU_PMU_AXI1_WR_DATA_BEAT_WRITTEN",
    "ETHOSU_PMU_AXI1_WR_TRAN_REQ_STALLED",
    "ETHOSU_PMU_AXI1_WR_DATA_BEAT_STALLED",
    "ETHOSU_PMU_AXI1_ENABLED_CYCLES",
    "ETHOSU_PMU_AXI1_RD_STALL_LIMIT",
    "ETHOSU_PMU_AXI1_WR_STALL_LIMIT",
    "ETHOSU_PMU_ECC_SB1",
    "ETHOSU_PMU_SENTINEL"
};




enum ethosu_pmu_event_type EVTYPES[4] = {
    //ETHOSU_PMU_MAC_ACTIVE,
    //ETHOSU_PMU_MAC_DPU_ACTIVE,
    //ETHOSU_PMU_AXI0_RD_TRANS_ACCEPTED,
    //ETHOSU_PMU_CC_STALLED_ON_BLOCKDEP
    ETHOSU_PMU_NPU_ACTIVE,
    ETHOSU_PMU_WD_ACTIVE,
    ETHOSU_PMU_MAC_ACTIVE,
    ETHOSU_PMU_AO_ACTIVE

};



uint32_t measure_layer0_start;
uint64_t start_cycles;
uint32_t start_vals [4];


//This gets called by ethosu_inference_begin() in ethosu_cpu_cache.c
void ethosu_start_pmu_measure(struct ethosu_driver *drv, void *user_arg)
{
    if (MEASURE_MODE) {
        // Start measuring time
        measure_layer0_start = debug_start_timer();

        // 1. Enable PMU
        ETHOSU_PMU_Enable(drv);


        // 2. Configure counters
        ETHOSU_PMU_Set_EVTYPER(drv, 0, EVTYPES[0]);
        ETHOSU_PMU_Set_EVTYPER(drv, 1, EVTYPES[1]);
        ETHOSU_PMU_Set_EVTYPER(drv, 2, EVTYPES[2]);
        ETHOSU_PMU_Set_EVTYPER(drv, 3, EVTYPES[3]);


        // 3. Reset counters
        ETHOSU_PMU_CYCCNT_Reset(drv);
        ETHOSU_PMU_EVCNTR_ALL_Reset(drv);

        uint32_t mask = ETHOSU_PMU_CCNT_Msk | ETHOSU_PMU_CNT1_Msk | ETHOSU_PMU_CNT2_Msk | ETHOSU_PMU_CNT3_Msk | ETHOSU_PMU_CNT4_Msk;
        ETHOSU_PMU_CNTR_Enable(drv, mask);

        start_cycles = ETHOSU_PMU_Get_CCNTR(drv);

        for (size_t i = 0; i < 4; i++ ){
            start_vals[i] = ETHOSU_PMU_Get_EVCNTR(drv, i);
        }

    }

}

void ethosu_inference_end(struct ethosu_driver *drv, void *user_arg)
{
    // Your custom end‑of‑inference hook
    // e.g., log results or stop the timer
    if (MEASURE_MODE) {
        uint32_t measure_layer0_elapsed_ticks = debug_end_timer(measure_layer0_start);

        uint64_t end_cycles = ETHOSU_PMU_Get_CCNTR(drv);

        uint32_t end_vals [4];
        for (size_t i = 0; i < 4; i++ ){
            end_vals[i] = ETHOSU_PMU_Get_EVCNTR(drv, i);
        }



        // 7. Disable counters
        for (size_t i = 0; i < 4; i++ ){
            ETHOSU_PMU_CNTR_Disable(drv, EVTYPES[i]);
        }


        // 10. Disable PMU if no longer needed
        ETHOSU_PMU_Disable(drv);

        // Log results
        uint64_t cycles = end_cycles - start_cycles;

        uint32_t measured_vals [4];
        for (size_t i = 0; i < 4; i++ ){
            measured_vals[i] = end_vals[i] - start_vals[i];
            printf("%s = %" PRIu32 "\n", event_type_names[EVTYPES[i]], measured_vals[i]);

        }

        // 9. Report or log
        //double utilization = (double)mac_active / (double)(mac_active + idle_cycles + block_stalls + axi_stalls);
        //printf("NPU_active + NPU_IDLE should be equal to NPU Cycles:%" PRIu32 "\n", mac_active+idle_cycles);
        printf("Npu cycles for it: %d = %" PRIu64 "\n", global_it, cycles);
        //printf("Npu MAC Active for it: %d = %" PRIu32 "\n", global_it, macs);
        ////printf("Npu MAC 8bit Active for it: %d = %" PRIu32 "\n", global_it, cc_dpu_active);
        //printf("Npu CCs where DPU is Active for it: %d = %" PRIu32 "\n", global_it, cc_dpu_active);
        //printf("Npu AXI0 Reads for it: %d = %" PRIu32 "\n", global_it, axi0_reads);
        //printf("Npu Blockdep Stalls for it: %d = %" PRIu32 "\n", global_it, blockdep_stalls);
        ////printf("Utilization: %f\n", utilization);
        ////printf("\tidle_cycles: %" PRIu32 "\n", idle_cycles);
        ////printf("\tblock_stalls: %" PRIu32 "\n", block_stalls);
        ////printf("\taxi_stalls: %" PRIu32 "\n", axi_stalls);

        //printf("Investigate start and end values:\n");
        //printf("\tNPU Cycles:\t\t\tstart: %" PRIu64 "\tend: %" PRIu64"\n", start_cycles, end_cycles);
        //printf("\tNPU MAC Operations:\t\tstart: %" PRIu32 "\tend: %" PRIu32"\n", start_mac, end_mac);
        ////printf("\tNPU MAC 8-bit Operations:\tstart: %" PRIu32 "\tend: %" PRIu32"\n", start_mac_8bit, end_mac_8bit);
        //printf("\tNPU CCs where DPU is active:\tstart: %" PRIu32 "\tend: %" PRIu32"\n", start_dpu, end_dpu);
        //printf("\tNPU AXI0 Reads:\t\t\tstart: %" PRIu32 "\tend: %" PRIu32"\n", start_axi0_read, end_axi0_read);
        //printf("\tNPU BlockDep Stalls:\t\tstart: %" PRIu32 "\tend: %" PRIu32"\n", start_blockdep_stalls, end_blockdep_stalls);


        // Print CPU timer results
        printf("Ticks elapsed for layer once in it: %d = %d \xC2\xB5s\n", global_it, measure_layer0_elapsed_ticks);
        //printf("Ticks elapsed for layer once in it: %d = %d ms\n", global_it, measure_layer0_elapsed_ticks);
        //printf("Ticks elapsed for layer once in it: %d = %" PRIu64 " \n", global_it, cycles);

    }
}






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








