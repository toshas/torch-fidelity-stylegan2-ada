#!/bin/bash
set -x
set -e

PATH_CIFAR10_TGZ="${HOME}/.cache/torch/fidelity_datasets/cifar-10-python.tar.gz"
DIR_CIFAR10=$(dirname "${PATH_CIFAR10_TGZ}")
if [ ! -f "${PATH_CIFAR10_TGZ}" ]; then
    mkdir -p "${DIR_CIFAR10}"
    cd "${DIR_CIFAR10}" && wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
fi

PATH_CIFAR10_ZIP="${DIR_CIFAR10}/cifar10.zip"
if [ ! -f "${PATH_CIFAR10_ZIP}" ]; then
    python dataset_tool.py --source="${PATH_CIFAR10_TGZ}" --dest="${PATH_CIFAR10_ZIP}"
fi

GPUS=4
G_CIFAR10=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

python calc_metrics.py --gpus=${GPUS} --metrics=\
ppl_zend_cifar10_original_epsexp_m4_dtype_u8,\
ppl_zend_cifar10_fidelity_epsexp_m4_dtype_u8 \
--data="${PATH_CIFAR10_ZIP}" --network="${G_CIFAR10}"
