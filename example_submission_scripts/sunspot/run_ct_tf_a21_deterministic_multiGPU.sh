#!/bin/bash -l
#PBS -l select=1:system=sunspot
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q workq-route
#PBS -A Aurora_deployment

# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/CosmicTagger
cd ${WORK_DIR}

DATA_DIR=/lus/gila/projects/Aurora_deployment/cadams/cosmic_tagger/

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=8

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=1
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo $GLOBAL_BATCH_SIZE


# For cosmic tagger, this improves performance:
# (for reference, the default is "setenv ITEX_LAYOUT_OPT \"1\" ")
unset ITEX_LAYOUT_OPT

export HOROVOD_LOG_LEVEL=INFO
export HOROVOD_CCL_FIN_THREADS=1
export HOROVOD_CCL_ADD_EXTRA_WAIT=1
export HOROVOD_FUSION_THRESHOLD=$((128*1024*1024))
export HOROVOD_CYCLE_TIME=0.1
unset HOROVOD_THREAD_AFFINITY

export CCL_LOG_LEVEL=WARN
export CCL_ZE_QUEUE_INDEX_OFFSET=0
export CCL_SYCL_OUTPUT_EVENT=0
export CCL_OP_SYNC=1
export CCL_USE_EXTERNAL_QUEUE=1
export CCL_ATL_TRANSPORT=mpi


# This is a fix for running over 16 nodes:
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OVFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20





LEARNING_RATE=0.03

# export NVIDIA_TF32_OVERRIDE=0
run_id=lr${LEARNING_RATE}/single-PVC-fp32-XPU-run2-hvd

# output_dir=/home/cadams/CosmicTagger/output/ \

# GPU Params:
COMPUTE_MODE=XPU
module load frameworks/2023-03-03-experimental
module list
source /home/cadams/frameworks-2023-01-31-extension/bin/activate
export NUMEXPR_MAX_THREADS=1



mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} \
--depth=8 --cpu-bind=verbose,depth \
python bin/exec.py \
--config-name a21-deterministic \
run.id=${run_id} \
run.distributed=True \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.compute_mode=${COMPUTE_MODE} \
data.data_directory=${DATA_DIR} \
mode.optimizer.loss_balance_scheme=focal \
mode.optimizer.learning_rate=${LEARNING_RATE}
