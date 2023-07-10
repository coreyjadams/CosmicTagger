#!/bin/bash -l
#PBS -l select=1:system=sunspot
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q workq
#PBS -A Aurora_deployment

# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/CosmicTagger
cd ${WORK_DIR}

DATA_DIR=/lus/gila/projects/Aurora_deployment/cadams/cosmic_tagger/

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=8
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo $GLOBAL_BATCH_SIZE


# export NVIDIA_TF32_OVERRIDE=0

# output_dir=/home/cadams/CosmicTagger/output/ \

# GPU Params:
COMPUTE_MODE=XPU
module load frameworks/2023-03-03-experimental
module list
source /home/cadams/frameworks-2023-01-31-extension/bin/activate
export NUMEXPR_MAX_THREADS=1


declare -a LEARNING_RATES=("0.003"  "0.003" "0.0003")
declare -a RUNS=("1" "2")
declare -a LOSSES=("none" "focal")

for LR in 0.03 0.003 0.0003;
do
  echo "$LR"
  for RUN in 1 2;
  do
    echo " $RUN"
    for LOSS in "none" "focal";
    do
      echo "  $LOSS"
      run_id=packed_lr${LR}/single-PVC-fp32-XPU-${LOSS}-run${RUN}
      echo $run_id
      python bin/exec.py \
      --config-name a21-deterministic \
      run.id=${run_id} \
      run.distributed=False \
      run.minibatch_size=${GLOBAL_BATCH_SIZE} \
      run.compute_mode=${COMPUTE_MODE} \
      data.data_directory=${DATA_DIR} \
      mode.optimizer.loss_balance_scheme=${LOSS} \
      mode.optimizer.learning_rate=${LR} \
      >/dev/null 2>&1 &

    done
  done
done
#

echo "Jobs launched"
wait < <(jobs -p)
