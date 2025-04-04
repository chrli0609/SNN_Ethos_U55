#pragma once



#include <stdint.h>










int membrane_update(
    uint8_t* v_mem,
    uint8_t* in_spk,
    const uint8_t* weight_tensor_ptr,
    float decay, //beta * num_time_steps_since_update
    float threshold,

    uint8_t* out_spk);

struct myTuple;

struct myTuple leaky_integrate_fire(float membrane_voltage, float x, float w, float beta, float threshold);

