

rm -r saved_models/*



SAVED_MODELS_DIR="saved_models"
TF_MODEL_DIR="${SAVED_MODELS_DIR}/TF_model"
TFLITE_DIR="${SAVED_MODELS_DIR}/tflite_model"
VELA_TFLITE_DIR="${SAVED_MODELS_DIR}/tflite_vela"


mkdir ${TF_MODEL_DIR}
mkdir ${TFLITE_DIR}
mkdir ${VELA_TFLITE_DIR}

TFLITE_PATH="${TFLITE_DIR}/tflite_model.tflite"

python3 LIF_model_tf.py ${TF_MODEL_DIR}


python3 tf_2_tflite.py ${TF_MODEL_DIR} ${TFLITE_PATH}



#compile with vela
vela ${TFLITE_PATH} --output-dir ${VELA_TFLITE_DIR} --verbose-operators --verbose-high-level-command-stream
#vela "saved_models/tflite_model/tflite_model.tflite" --output-dir "saved_models/tflite_vela" --verbose-operators --verbose-high-level-command-stream


echo "Illustrate Vela Compiled Model in Netron? (y/n)"

read userVal

if [ $userVal == "y" ]; then
	netron saved_models/tflite_vela/tflite_model_vela.tflite
fi
