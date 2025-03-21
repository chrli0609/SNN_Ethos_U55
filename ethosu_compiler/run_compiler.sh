#!/bin/bash

#IN_FILEPATH="command_stream/conv2d.txt"
#OUT_FILEPATH="command_stream/conv2d_translated.hpp"

IN_FILEPATH=$1
IN_FILENAME=$(basename ${IN_FILEPATH})
OUT_FILEPATH="output/${IN_FILENAME%.*}_translated.hpp"

python3 src/my_translator.py "${IN_FILEPATH}" "${OUT_FILEPATH}"



