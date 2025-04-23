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














// Inputs:
//  * v_mem, dim=(LAYER i)
//  * in_spk, dim=(layer i-1)
//  * weights, dim=(depends on encoding)
//  * beta, scalar
//  * threshold, scalar
// Output:
//  * v_mem, dim=(layer i)
//  * out_spk, dim=(layer i)

