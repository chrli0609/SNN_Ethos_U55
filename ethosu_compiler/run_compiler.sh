#!/bin/bash


#Usage: ./run_compiler <path/to/cms_file> <path/to/weight_params_file>

IN_FILEPATH=$1
IN_FILENAME=$(basename ${IN_FILEPATH})
OUT_FILEPATH="output/${IN_FILENAME%.*}_translated.hpp"
WEIGHT_PARAMS_FILE=$2

python3 src/main.py "${IN_FILEPATH}" "${WEIGHT_PARAMS_FILE}" "${OUT_FILEPATH}"



