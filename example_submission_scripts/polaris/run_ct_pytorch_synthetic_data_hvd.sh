#!/bin/sh
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:15:00
#PBS -q debug-scaling
#PBS -l filesystems=home
#PBS -A datascience

# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/khalid_CT/CosmicTagger
cd ${WORK_DIR}

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=1

# Set up software deps:
module load conda/2022-09-08
conda activate


mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
python bin/exec.py \
--config-name polaris \
data=synthetic \
run.id=polaris_${LOCAL_BATCH_SIZE}-ranks${NRANKS}-nodes${NNODES} \
run.distributed=True \
run.minibatch_size=${LOCAL_BATCH_SIZE} \
run.iterations=500 \
framework.distributed_mode=horovod 
