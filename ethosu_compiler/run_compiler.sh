#!/bin/bash


#Usage: ./run_compiler <model_name>
#Note: model is stored under gen_cms


cd gen_cms/
python3 main.py --model "${1}"
