from pathlib import Path
import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


current_working_directory = Path(os.getcwd())
current_to_project_directory = Path("../../../snn_on_alif_e7/simple_code_test/")

import layer0
import layer1


if len(sys.argv) > 1:
        try:
            layer0_OUTPUT_LAYER_SIZE = int(sys.argv[1])
        except:
            print("Expected Integer command line argument but received:", sys.argv[1])
            exit()
else:
    print("AN ERROR HAS OCCURRRED, INCORRECT COMMAND LINE ARGUMENTS SET WHEN CALLING main.py")


LAYER_0_CMS_NAME = "fc_lif_layer_0"
LAYER_1_CMS_NAME = "fc_lif_layer_1"


#header_out_filepath_layer0 = current_working_directory + '/' + "../../../snn_on_alif_e7/simple_code_test/include/" + LAYER_0_CMS_NAME + ".h"
#imp_out_filepath_layer0 = current_working_directory + '/' + "../../../snn_on_alif_e7/simple_code_test/nn_ops/" + LAYER_0_CMS_NAME + ".c"

header_out_filepath_layer0 = current_working_directory / current_to_project_directory / Path("include") / Path(LAYER_0_CMS_NAME+ ".h")
imp_out_filepath_layer0 = current_working_directory / current_to_project_directory / Path("nn_ops") / Path(LAYER_0_CMS_NAME + ".c")


layer0.main(OUTPUT_LAYER_SIZE=layer0_OUTPUT_LAYER_SIZE, cms_name=LAYER_0_CMS_NAME, header_out_filepath=header_out_filepath_layer0, imp_out_filepath=imp_out_filepath_layer0)


header_out_filepath_layer1 = current_working_directory / current_to_project_directory / Path("include") / Path(LAYER_1_CMS_NAME + ".h")
imp_out_filepath_layer1 = current_working_directory / current_to_project_directory / Path("nn_ops") / Path(LAYER_1_CMS_NAME + ".c")

layer1.layer1_merge_and_write(cms_name=LAYER_1_CMS_NAME, header_out_filepath=header_out_filepath_layer1, imp_out_filepath=imp_out_filepath_layer1)





# Notes
'''
*   In_spk scale and zero point probably should match out_spk if
    they are to be connected
'''
