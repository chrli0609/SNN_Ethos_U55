#!/bin/bash
set -x


SSH_PASSWORD="TO_BE_SET"
USR_NAME="TO_BE_SET"


rm -rf "${1}"/model_params
rm -rf "${1}"/test_patterns



sshpass -p "$SSH_PASSWORD" scp -r ${USR_NAME}@172.16.222.25:/home/${USR_NAME}/n_mnist_qat_snn/"${1}"/model_params "${1}"
sshpass -p "$SSH_PASSWORD" scp -r ${USR_NAME}@172.16.222.25:/home/${USR_NAME}/n_mnist_qat_snn/"${1}"/test_patterns "${1}" 

