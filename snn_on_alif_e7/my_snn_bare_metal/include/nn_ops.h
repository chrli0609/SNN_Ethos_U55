#pragma once



#include <stdint.h>
#include <stddef.h>








#include <ethosu_driver.h>
void ethosu_start_pmu_measure(struct ethosu_driver *drv, void *user_arg);




int run_cms(
    const uint8_t* command_stream,
    size_t command_stream_size,
    uint64_t* base_addrs,
    size_t* base_addrs_size,
    int num_tensors);
