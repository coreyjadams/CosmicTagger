#!/bin/bash -l
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l walltime=3:00:00
#PBS -q workq
#PBS -A datascience

# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger
cd ${WORK_DIR}



# How many nodes?
NNODES=`wc -l < $PBS_NODEFILE`
# On Polaris, 4 ranks per node:
NRANKS_PER_NODE=4
let NRANKS=${NNODES}*${NRANKS_PER_NODE}

# What to use for local batch size?
# Depends on framework, downsampling and model.
# Suggestions: 
# - Tensorflow takes about 2x more memory than torch
# - A21 model has many less layers, can fit more images per batch
# - Uresnet model struggles to fit many places



# Set parameters:
MODEL=a21 # a21 or uresnet (or maybe others)
FRAMEWORK=torch # torch or tensorflow
PRECISION=mixed # float32, mixed, or bfloat16
DISTRIBUTED_MODE=DDP # horovod or DDP, but only for torch
DOWNSAMPLE=1 # 0, 1, 2, 3 ... but if you stray from 0/1/2 you will need to adjust the model depth

# Data parameters:
DATA=real # or, synthetic

# Where is the data hosted?  v1 only on the master branch:
DATA_DIR=/lus/eagle/projects/datasets/CosmicTagger/v1/

DATA_ARGS="data=${DATA} data.downsample=${DOWNSAMPLE}"

if [ "$DATA" == "real" ]; then
    DATA_ARGS="${DATA_ARGS} data.data_directory=${DATA_DIR} "
fi

echo $DATA_ARGS

FRAMEWORK_ARGS="framework=${FRAMEWORK}"
if [ "${FRAMEWORK}" == "torch" ]; then
    FRAMEWORK_ARGS="${FRAMEWORK_ARGS} framework.distributed_mode=${DISTRIBUTED_MODE} "
fi

echo ${FRAMEWORK_ARGS}

LOCAL_BATCH_SIZE=8
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo "Global Batch Size: $GLOBAL_BATCH_SIZE"

# Set up software deps:
module use /soft/modulefiles/
module load conda/2024-04-29; conda activate

# Add-ons from conda:
source /home/cadams/Polaris/polaris_conda_2024-04-29-venv/bin/activate

# A name for this run:
run_id=cosmic_tagger_real_data-${MODEL}-${FRAMEWORK}-${DISTRIBUTED_MODE}-B${GLOBAL_BATCH_SIZE}-R${NRANKS}

# TF Env Variables:
TF_USE_LEGACY_KERAS=1

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
python bin/exec.py \
--config-name ${MODEL} \
${DATA_ARGS} \
$FRAMEWORK_ARGS \
run.id=${run_id} \
run.distributed=True  \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.precision=${PRECISION} \
run.compute_mode=GPU \
mode.optimizer.loss_balance_scheme=light \
run.iterations=100
