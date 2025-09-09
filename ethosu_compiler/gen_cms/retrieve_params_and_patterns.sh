#!/bin/bash
set -x


## TO BE SET ##
SSH_ADDR=""
SSH_PASSWORD=""  
SSH_USRNAME=""

rm -rf "${1}"/model_params
rm -rf "${1}"/test_patterns



sshpass -p "$SSH_PASSWORD" scp -r ${SSH_USRNAME}@${SSH_ADDR}:/home/${SSH_USRNAME}/n_mnist_qat_snn/"${1}"/model_params "${1}"
sshpass -p "$SSH_PASSWORD" scp -r ${SSH_USRNAME}@${SSH_ADDR}:/home/${SSH_USRNAME}/n_mnist_qat_snn/"${1}"/test_patterns "${1}" 

