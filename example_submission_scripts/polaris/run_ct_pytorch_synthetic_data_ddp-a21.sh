#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A datascience

# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger/
cd ${WORK_DIR}

OUTPUT_DIR=/home/cadams/Polaris/CT_Scaling_Maintenance/pt-a21/
# OUTPUT_DIR=/lus/grand/projects/datascience/cadams/CT-Scaling-Tests/ddp/

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

echo "NRANKS is ${NRANKS}"

LOCAL_BATCH_SIZE=8


# Set up software deps:

module load conda/2022-09-08
conda activate

LAUNCH_TIME=$(date +"%F:%H:%M")

RUNID=polaris_${LOCAL_BATCH_SIZE}-ranks${NRANKS}-nodes${NNODES}-lt${LAUNCH_TIME}-fp32

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
# export NVIDIA_TF32_OVERRIDE=0

export TF_XLA_FLAGS=--tf_xla_auto_jit=2

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
python bin/exec.py \
--config-name a21 \
mode=iotest  \
run.id=${RUNID} \
run.minibatch_size=${LOCAL_BATCH_SIZE} \
run.iterations=102

# run.compute_mode=GPU \
# framework=tensorflow \
# data=synthetic \
# output_dir=${OUTPUT_DIR}/${RUNID} \
# run.precision=float32 \
# run.distributed=False \

# FP32 Single Device / Full Node:
# TF - 9.5369 / 35.2517 (92.4%)
# TF XLA - 21.9701 / 64.0149 (72.84%)
# Torch - 14.7437 / 54.8887 (93.07%)


# TF32 Single Device:
# TF - 11.7887 / 39.1704 (83.1%)
# TF XLA - 29.1168 / 78.3040 (67.232%)
# Torch - 15.4160 / 57.2924 (92.9%)

# Mixed FP16 Single Device:
# TF - 11.3194 / 40.8954 (90.32%)
# TF XLA - 38.9021 / 98.4070 (63.2%)
# Torch - 15.3899 / 57.0449 (92.666%)

# AMD Mi250 FP32
# TF - 
# TF XLA - 
# Torch - 15.4053 / 119.5214 (97%)

# AMD Mi250 FP16 mixed
# Torch -  14.5396 / 114.8543 ()