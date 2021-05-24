#!/bin/bash
set -x
set -e

GPUS=4
DATA_CIFAR10=/raid/${USER}/datasets/torchvision/cifar10
G_CIFAR10=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

python dataset_tool.py --source=${DATA_CIFAR10}/cifar-10-python.tar.gz --dest=${DATA_CIFAR10}/cifar10.zip

python calc_metrics.py --gpus=${GPUS} --metrics=\
ppl_zend_cifar10_original_epsexp_m4_dtype_u8,\
ppl_zend_cifar10_fidelity_epsexp_m4_dtype_u8,\
ppl_zend_cifar10_original_epsexp_m4_dtype_f32,\
ppl_zend_cifar10_fidelity_epsexp_m4_dtype_f32,\
ppl_zend_cifar10_original_epsexp_m3_dtype_u8,\
ppl_zend_cifar10_fidelity_epsexp_m3_dtype_u8,\
ppl_zend_cifar10_original_epsexp_m3_dtype_f32,\
ppl_zend_cifar10_fidelity_epsexp_m3_dtype_f32,\
ppl_zend_cifar10_original_epsexp_m2_dtype_u8,\
ppl_zend_cifar10_fidelity_epsexp_m2_dtype_u8,\
ppl_zend_cifar10_original_epsexp_m2_dtype_f32,\
ppl_zend_cifar10_fidelity_epsexp_m2_dtype_f32,\
ppl_zend_cifar10_original_epsexp_m1_dtype_u8,\
ppl_zend_cifar10_fidelity_epsexp_m1_dtype_u8,\
ppl_zend_cifar10_original_epsexp_m1_dtype_f32,\
ppl_zend_cifar10_fidelity_epsexp_m1_dtype_f32 \
--data=${DATA_CIFAR10}/cifar10.zip --network=${G_CIFAR10}
