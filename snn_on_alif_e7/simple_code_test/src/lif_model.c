//typedef int bool;
//#define true 1
//#define false 0




struct myTuple {
    int spike;
    float membrane_voltage;
};



struct myTuple leaky_integrate_fire(float membrane_voltage, float x, float w, float beta, float threshold) {

    int spike;
    if (membrane_voltage > threshold) {
        spike = 1;
    } else {
        spike = 0;
    }

    float new_membrane_voltage = beta * membrane_voltage + w * x - spike * threshold;


    struct myTuple r = {spike, new_membrane_voltage};

    return r;
}