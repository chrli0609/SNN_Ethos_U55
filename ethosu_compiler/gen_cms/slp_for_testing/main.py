from pathlib import Path
import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


current_working_directory = Path(os.getcwd())
current_to_model_directory = Path("../../../snn_on_alif_e7/my_snn_bare_metal/nn_models/")



'''
Command line argument: LAYER_0_OUTPUT_SIZE
'''

MODEL_NAME = "multi_tensor_sram_mlp"
LAYER_0_CMS_NAME = "fc_lif_layer_0"
LAYER_1_CMS_NAME = "fc_lif_layer_1"



import layer0
import layer1


if len(sys.argv) > 1:
        try:
            layer0_INPUT_LAYER_SIZE = int(sys.argv[1])
            layer0_OUTPUT_LAYER_SIZE = int(sys.argv[2])
        except:
            print("Expected Integer command line argument but received:", sys.argv[1], "and", sys.argv[2])
            exit()
else:
    print("AN ERROR HAS OCCURRRED, INCORRECT COMMAND LINE ARGUMENTS SET WHEN CALLING main.py")







# Assign header filenames
header_out_filepath_layer0 = current_working_directory / current_to_model_directory / Path(MODEL_NAME) / Path("layers") / Path(LAYER_0_CMS_NAME+ ".h")
header_out_filepath_layer1 = current_working_directory / current_to_model_directory / Path(MODEL_NAME) / Path("layers") / Path(LAYER_1_CMS_NAME + ".h")



layer0.main(INPUT_LAYER_SIZE=layer0_INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE=layer0_OUTPUT_LAYER_SIZE, cms_name=LAYER_0_CMS_NAME, header_out_filepath=header_out_filepath_layer0)
#layer1.layer1_merge_and_write(cms_name=LAYER_1_CMS_NAME, header_out_filepath=header_out_filepath_layer1)
layer1.main(cms_name=LAYER_1_CMS_NAME, header_out_filepath=header_out_filepath_layer1)





# Notes
'''
*   In_spk scale and zero point probably should match out_spk if
    they are to be connected
'''
