#include "include/nn_ops.h"






#include <stdio.h>
#include <stddef.h>




#include "ethosu_driver.h"
//#include "include/extra_funcs.h"







//mydebug
#include "include/extra_funcs.h"    //SysTick_Handler and start and end timer
extern int DEBUG_MODE;
extern int MEASURE_MODE;
extern int global_it;













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


    uint32_t measure_layer0_start;
    if (MEASURE_MODE) {
        measure_layer0_start = debug_start_timer();
    }

    if(ethosu_invoke_v3(drv, command_stream, command_stream_size, 
        base_addrs, base_addrs_size, num_tensors, NULL) != 0) {
        printf("Invoke_v3 Failed\n");
        return -1;
    }
    if (MEASURE_MODE) {
        uint32_t measure_layer0_elapsed_ticks = debug_end_timer(measure_layer0_start);
        printf("Ticks elapsed for layer once in it: %d = %d\n", global_it, measure_layer0_elapsed_ticks);
    }

    //mydebug
    if (DEBUG_MODE) { printf("After invoke_v3: &drv = %p, drv = %p\n", (void*)&drv, (void*)drv); }


    ethosu_release_driver(drv);

    if (DEBUG_MODE) { printf("Driver release successfully\n"); }



    return 0;

}








