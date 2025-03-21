

rm -r saved_models/*



SAVED_MODELS_DIR="saved_models"
TF_MODEL_DIR="${SAVED_MODELS_DIR}/TF_model"
TFLITE_DIR="${SAVED_MODELS_DIR}/tflite_model"
VELA_TFLITE_DIR="${SAVED_MODELS_DIR}/tflite_vela"


mkdir ${TF_MODEL_DIR}
mkdir ${TFLITE_DIR}
mkdir ${VELA_TFLITE_DIR}

TFLITE_PATH="${TFLITE_DIR}/tflite_model.tflite"

python3 simple_model.py ${TF_MODEL_DIR}


python3 tf_2_tflite.py ${TF_MODEL_DIR} ${TFLITE_PATH}



#compile with vela
vela ${TFLITE_PATH} --output-dir ${VELA_TFLITE_DIR} \
	--verbose-operators \
	--verbose-high-level-command-stream \
	--verbose-register-command-stream \
	--verbose-weights \
	--verbose-allocation \
	--verbose-tensor-format \
	--verbose-tensor-purpose \
	--verbose-operators \
	--verbose-config


echo "Illustrate Vela Compiled Model in Netron? (y/n)"

read userVal

#if [ $userVal == "y" ]; then
#	netron saved_models/tflite_vela/tflite_model_vela.tflite
#fi


cd ../

python3 gen_model_cpp.py --tflite_path conv2d_test/saved_models/tflite_vela/tflite_model_vela.tflite --output_dir conv2d_test/saved_models/cpp_tflite_vela




