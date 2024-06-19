#!/bin/bash -l
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l walltime=3:00:00
#PBS -q workq
#PBS -A datascience

# Set parameters.  Check if they are already set, if not use defaults.  You can always change defaults, too
if [[ -z "$MODEL" ]]; then
    MODEL=a21 # a21 or uresnet (or maybe others)
fi

if [[ -z "FRAMEWORK" ]]; then
	FRAMEWORK=tensorflow # torch or tensorflow
fi

if [[ -z "$PRECISION" ]]; then
	PRECISION=mixed # float32, mixed, or bfloat16
fi

if [[ -z "$DISTRIBUTED_MODE" ]]; then
	DISTRIBUTED_MODE=DDP # horovod or DDP, but only for torch
fi

if [[ -z "$DOWNSAMPLE" ]]; then 
	DOWNSAMPLE=1 # 0, 1, 2, 3 ... but if you stray from 0/1/2 you will need to adjust the model depth
fi

echo "FRAMEWORK: ${FRAMEWORK}"
echo "PRECISION: ${PRECISION}"
echo "DOWNSAMPLE: ${DOWNSAMPLE}"

return 
# What to use for local batch size?
# Depends on framework, downsampling and model.
# Suggestions: 
# - Tensorflow takes about 2x more memory than torch
# - A21 model has many less layers, can fit more images per batch
# - Uresnet model struggles to fit many places
LOCAL_BATCH_SIZE=8


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger
cd ${WORK_DIR}
OUTPUT_DIR=/lus/eagle/projects/datascience/cadams/test_ct_output_junk

# Detect the system and set up environmet and data:
if [[ $(hostname -f) == *"polaris"* ]]; 
then 
    echo "Set up for Polaris"; 
    NRANKS_PER_NODE=4

    # Where is the data hosted?  v1 only on the master branch:
    DATA_DIR=/lus/eagle/projects/datasets/CosmicTagger/v1/

    # Set up software deps:
    module use /soft/modulefiles/
    module load conda/2024-04-29; conda activate

    # Add-ons from conda:
    source /home/cadams/Polaris/polaris_conda_2024-04-29-venv/bin/activate

elif [[ $(hostname -f) == *"aurora"* ]];
then
    echo "Set up for Aurora";
    NRANKS_PER_NODE=12
    DATA_DIR=MISSING

elif [[ $(hostname -f) == *"americas"* ]];
then
    echo "Set up for Sunspot"
    NRANKS_PER_NODE=12
    DATA_DIR=MISSING
else
    echo "Why did Corey leave? Now my code doesn't work :'("
fi


# How many nodes?
NNODES=`wc -l < $PBS_NODEFILE`
# On Polaris, 4 ranks per node:
let NRANKS=${NNODES}*${NRANKS_PER_NODE}




# Data parameters:
DATA=real # or, synthetic

DATA_ARGS="data=${DATA} data.downsample=${DOWNSAMPLE}"

if [ "$DATA" == "real" ]; then
    DATA_ARGS="${DATA_ARGS} data.data_directory=${DATA_DIR} "
fi


FRAMEWORK_ARGS="framework=${FRAMEWORK}"
if [ "${FRAMEWORK}" == "torch" ]; then
    FRAMEWORK_ARGS="${FRAMEWORK_ARGS} framework.distributed_mode=${DISTRIBUTED_MODE} "
fi


let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo "Global Batch Size: $GLOBAL_BATCH_SIZE"


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
run.iterations=100 \
output_dir=${OUTPUT_DIR}
