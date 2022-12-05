#!/bin/sh
#PBS -l select=32:system=polaris
#PBS -l place=scatter
#PBS -l walltime=3:30:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=home:grand


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=2
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

# Set up software deps:
module load conda/2022-09-08
conda activate

# Add-ons from conda:
source /home/cadams/Polaris/polaris_conda_2022-09-08-venv/bin/activate

module load cray-hdf5/1.12.1.3

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=none \
python bin/exec.py \
--config-name uresnet2 \
run.id=eventID-256-Vertex-256-vd0-${GLOBAL_BATCH_SIZE}-2 \
data.downsample=1 \
run.distributed=True \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.iterations=20000 \
network.depth=6 \
network.vertex.detach=True \
network.vertex.depth=1 \
network.vertex.n_filters=256 \
network.classification.detach=True \
network.classification.n_filters=256 \
network.n_initial_filters=64 \
framework=torch
