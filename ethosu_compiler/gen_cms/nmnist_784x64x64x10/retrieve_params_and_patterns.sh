#!/bin/bash
set -x


SSH_PASSWORD="1qa2ws3ed"


rm -rf model_params
rm -rf test_patterns



sshpass -p "$SSH_PASSWORD" scp -r chrliu@172.16.222.25:/home/chrliu/n_mnist_qat_snn/model_params . 1>&1 2>&2
sshpass -p "$SSH_PASSWORD" scp -r chrliu@172.16.222.25:/home/chrliu/n_mnist_qat_snn/test_patterns . 

