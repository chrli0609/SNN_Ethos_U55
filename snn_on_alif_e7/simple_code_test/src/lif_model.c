#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h> //malloc?
#include <stdint.h>



#include "include/lif_model.h"
#include "include/nn_ops.h"
#include "include/extra_funcs.h"


#include "include/my_mem_u.h"










int my_mem_update(
    NN_Model* mlp_model,

    float* in_spk,

    float* ln_beta,
    float* vth,
    float* v_mem,
    float* time_not_updated,

    float* out_spk
) {

        #include "include/init_nn_model.h"
        MLP_Quantize_Inputs(mlp_model, in_spk, ln_beta, vth, v_mem, time_not_updated);




        #include "include/nn_data_structure.h"
        // First layer
        NNLayer* nnlayer = mlp_model->first_nnlayer;

        // Check Tensor Arena Values Before NPU OP
        NNLayer_DequantizeAndPrint(nnlayer);




        // Run NPU Membrane Update
        my_mem_u_npu(
            nnlayer->tensor_arena,
            nnlayer->tensor_arena_size,

            Getmy_mem_uCMSPointer(),
            Getmy_mem_uCMSLen(),
            Getmy_mem_uWeightsPointer(),
            Getmy_mem_uWeightsLen(),


            Getmy_mem_uLUTPointer(),
            Getmy_mem_uLUTLen()
        );



        
        // Check resulting Tensor Arena Values after NPU OP
        NNLayer_DequantizeAndPrint(nnlayer);

}





// Inputs:
//  * v_mem, dim=(LAYER i)
//  * in_spk, dim=(layer i-1)
//  * weights, dim=(depends on encoding)
//  * beta, scalar
//  * threshold, scalar
// Output:
//  * v_mem, dim=(layer i)
//  * out_spk, dim=(layer i)

