#!/bin/bash -l
#PBS -l select=6:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home

# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger2
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NDEPTH=8

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=1
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

# Add-ons from conda:
module load conda
conda activate
source /home/cadams/Polaris/polaris_conda_2022-09-08-venv/bin/activate

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
python bin/exec.py \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.distributed=True \
framework=tensorflow \
run.precision=float32 \
run.id=convergence_${GLOBAL_BATCH_SIZE}_${NNODES}-2 \
run.run_length=1500 \
run.run_units=iteration 
