#!/bin/bash -l


#####################################################################
# These are my own personal directories,
# you will need to change these.
#####################################################################
OUTPUT_DIR_TOP=/lus/grand/projects/datascience/cadams/CT-single-gpu-perf/${DATE}
WORKDIR=/home/cadams/Polaris/CosmicTagger/
cd ${WORKDIR}

#####################################################################
# This block configures the total number of ranks, discovering
# it from PBS variables.
# 12 Ranks per node, if doing rank/tile
#####################################################################

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
let NRANKS=${NNODES}*${NRANKS_PER_NODE}

#####################################################################
# APPLICATION Variables that make a performance difference for torch:
#####################################################################

# Precision for CT can be float32, bfloat16, or mixed (fp16).
PRECISION="float32"
# PRECISION="bfloat16"
# PRECISION="mixed"

# Adjust the local batch size:
LOCAL_BATCH_SIZE=8

# NOTE: batch size 8 works ok, batch size 16 core dumps, haven't explored
# much in between.  reduced precision should improve memory usage.

#####################################################################
# End of perf-adjustment section
#####################################################################


module load conda/2022-09-08
conda activate

# Create the command arguments:

echo "CREATE ARGS"

PYTHON_ARGUMENTS="bin/exec.py --config-name a21 framework=${FRAMEWORK} "
PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.compute_mode=GPU run.distributed=False "
PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.precision=${PRECISION} run.minibatch_size=${LOCAL_BATCH_SIZE} run.iterations=500"
PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} mode.checkpoint_iteration=20000"

echo $PYTHON_ARGUMENTS


# One tile per gpu:
# ZE_AFFINITY_MASK=0.0
# ZE_AFFINITY_MASK=0.0,1.0,2.0,3.0,4.0,5.0
# ZE_AFFINITY_MASK=0.0,2.0,4.0
# Set up the environment:

#####################################################################
# End of environment setup section
#####################################################################

#####################################################################
# JOB LAUNCH
# Note that this example targets a SINGLE TILE
#####################################################################

# These are the tiles in order in which we use them:
# ZE_AFFINITY_LIST=("0.0" "1.0"  "2.0" "3.0" "4.0" "5.0" "0.1" "1.1" "2.1" "3.1" "4.1" "5.1")
# CPU_AFFINITY_LIST=("0"  "16"   "32"  "52"  "68"  "84"  "8"   "24"  "40"  "60"  "76"  "92")
CUDA_AFFINITY_LIST=("0" "1"  "2" "3")
CPU_AFFINITY_LIST=("24"  "16"   "8"  "0")
RUN_ID_TEMPLATE=polaris-a21-single-gpu-${FRAMEWORK}-n${NRANKS}-df${DATA_FORMAT}-p${PRECISION}-mb${LOCAL_BATCH_SIZE}-synthetic-gpu-run${RUN}


for (( idx=0; idx<${NRANKS_PER_NODE}; idx++ ))
do
	echo "Hello"
	echo $idx
	GPU=${CUDA_AFFINITY_LIST[$idx]}
	CPU=${CPU_AFFINITY_LIST[$idx]}
	host=$(hostname)
	run_id="${RUN_ID_TEMPLATE}/${host}/GPU${GPU}-CPU${CPU}"
	echo $run_id
	export CUDA_VISIBLE_DEVICES=${GPU}
	numactl -C ${CPU} python $PYTHON_ARGUMENTS run.id=${run_id} output_dir=${OUTPUT_DIR_TOP}/${run_id} > /dev/null 2>&1 &
	unset CUDA_VISIBLE_DEVICES
done

wait < <(jobs -p)

echo "DONE LOOP"

# This string is an identified to store log files:
