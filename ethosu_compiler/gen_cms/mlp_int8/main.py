
import os
current_working_directory = os.getcwd()


from layer0 import layer0_merge_and_write
from layer1 import layer1_merge_and_write

LAYER_0_CMS_NAME = "fc_lif_layer_0"
LAYER_1_CMS_NAME = "fc_lif_layer_1"

header_out_filepath_layer0 = current_working_directory + '/' + "../../../snn_on_alif_e7/simple_code_test/include/" + LAYER_0_CMS_NAME + ".h"
imp_out_filepath_layer0 = current_working_directory + '/' + "../../../snn_on_alif_e7/simple_code_test/nn_ops/" + LAYER_0_CMS_NAME + ".c"

layer0_merge_and_write(cms_name=LAYER_0_CMS_NAME, header_out_filepath=header_out_filepath_layer0, imp_out_filepath=imp_out_filepath_layer0)


header_out_filepath_layer1 = current_working_directory + '/' + "../../../snn_on_alif_e7/simple_code_test/include/" + LAYER_1_CMS_NAME + ".h"
imp_out_filepath_layer1 = current_working_directory + '/' + "../../../snn_on_alif_e7/simple_code_test/nn_ops/" + LAYER_1_CMS_NAME + ".c"

layer1_merge_and_write(cms_name=LAYER_1_CMS_NAME, header_out_filepath=header_out_filepath_layer1, imp_out_filepath=imp_out_filepath_layer1)


# Notes
'''
*   In_spk scale and zero point probably should match out_spk if
    they are to be connected
'''