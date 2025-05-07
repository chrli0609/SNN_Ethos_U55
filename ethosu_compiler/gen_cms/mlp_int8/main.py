from pathlib import Path
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


current_working_directory = Path(os.getcwd())
current_to_project_directory = Path("../../../snn_on_alif_e7/simple_code_test/")


from layer0 import layer0_merge_and_write
from layer1 import layer1_merge_and_write

LAYER_0_CMS_NAME = "fc_lif_layer_0"
LAYER_1_CMS_NAME = "fc_lif_layer_1"


#header_out_filepath_layer0 = current_working_directory + '/' + "../../../snn_on_alif_e7/simple_code_test/include/" + LAYER_0_CMS_NAME + ".h"
#imp_out_filepath_layer0 = current_working_directory + '/' + "../../../snn_on_alif_e7/simple_code_test/nn_ops/" + LAYER_0_CMS_NAME + ".c"

header_out_filepath_layer0 = current_working_directory / current_to_project_directory / Path("include") / Path(LAYER_0_CMS_NAME+ ".h")
imp_out_filepath_layer0 = current_working_directory / current_to_project_directory / Path("nn_ops") / Path(LAYER_0_CMS_NAME + ".c")

print("header_out_filepath:", header_out_filepath_layer0)
print("imp_out_filepath:", imp_out_filepath_layer0)

layer0_merge_and_write(cms_name=LAYER_0_CMS_NAME, header_out_filepath=header_out_filepath_layer0, imp_out_filepath=imp_out_filepath_layer0)


header_out_filepath_layer1 = current_working_directory / current_to_project_directory / Path("include") / Path(LAYER_1_CMS_NAME + ".h")
imp_out_filepath_layer1 = current_working_directory / current_to_project_directory / Path("nn_ops") / Path(LAYER_1_CMS_NAME + ".c")

layer1_merge_and_write(cms_name=LAYER_1_CMS_NAME, header_out_filepath=header_out_filepath_layer1, imp_out_filepath=imp_out_filepath_layer1)





# Notes
'''
*   In_spk scale and zero point probably should match out_spk if
    they are to be connected
'''
