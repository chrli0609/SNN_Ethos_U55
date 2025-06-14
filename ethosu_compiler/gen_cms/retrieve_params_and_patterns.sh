#!/bin/bash
set -x


SSH_PASSWORD="1qa2ws3ed"


rm -rf "${1}"/model_params
rm -rf "${1}"/test_patterns



sshpass -p "$SSH_PASSWORD" scp -r chrliu@172.16.222.25:/home/chrliu/n_mnist_qat_snn/"${1}"/model_params "${1}"
sshpass -p "$SSH_PASSWORD" scp -r chrliu@172.16.222.25:/home/chrliu/n_mnist_qat_snn/"${1}"/test_patterns "${1}" 

