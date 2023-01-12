#!/bin/sh
#PBS -l select=256:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q SDL_Workshop
#PBS -A SDL_Workshop


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NDEPTH=8

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

#With the synthetic data, the batch size parameter represents
#  the LOCAL batch size:

LOCAL_BATCH_SIZE=1
# Do this with REAL data:
# let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}
# Do this with synthetic data:
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}

echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

# Set up software deps:
module load conda/2022-07-19
conda activate

# Add-ons from conda:
# This isn't needed with synthetic data:
# source /home/cadams/Polaris/polaris_conda_2022-07-19-venv/bin/activate

module load cray-hdf5/1.12.1.3

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB

# export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind=depth \
python bin/exec.py \
--config-name a21 \
run.distributed=True \
run.id=a21_scaling_test_${GLOBAL_BATCH_SIZE}_${NNODES} \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.iterations=500 \
data.downsample=1