#!/bin/sh
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:grand


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=1
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

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
python bin/exec.py \
--config-name uresnet2 \
run.id=uresnet2-lightning-vd5-ds1-${GLOBAL_BATCH_SIZE} \
output_dir=/lus/grand/projects/datascience/cadams/CosmicTaggerVertexEventID/ \
data.downsample=1 \
data.data_format=channels_first \
run.distributed=True \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.run_length=1000 \
run.run_units=iteration \
network.depth=6 \
network.classification.detach=True \
network.n_initial_filters=32 \
network.vertex.active=False \
framework=lightning


# network.vertex.detach=True \
# network.vertex.depth=5 \
