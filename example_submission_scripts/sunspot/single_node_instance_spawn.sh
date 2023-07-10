#!/bin/bash -l


#####################################################################
# These are my own personal directories,
# you will need to change these.
#####################################################################
OUTPUT_DIR_TOP=/lus/gila/projects/Aurora_deployment/cadams/ct_output_multirun/${DATE}
WORKDIR=/home/cadams/CosmicTagger/
cd ${WORKDIR}

#####################################################################
# This block configures the total number of ranks, discovering
# it from PBS variables.
# 12 Ranks per node, if doing rank/tile
#####################################################################

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12
let NRANKS=${NNODES}*${NRANKS_PER_NODE}

#####################################################################
# APPLICATION Variables that make a performance difference for torch:
#####################################################################

# Channels last is faster for pytorch, requires code changes!
# More info here:
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/features.html#channels-last
DATA_FORMAT="channels_last"
# DATA_FORMAT="channels_first"


# Precision for CT can be float32, bfloat16, or mixed (fp16).
PRECISION="float32"
# PRECISION="bfloat16"
# PRECISION="mixed"

# Adjust the local batch size:
LOCAL_BATCH_SIZE=12

# NOTE: batch size 8 works ok, batch size 16 core dumps, haven't explored
# much in between.  reduced precision should improve memory usage.

#####################################################################
# FRAMEWORK Variables that make a performance difference for tf:
#####################################################################

# Toggle tf32 on (or don't):
IPEX_FP32_MATH_MODE=TF32
# unset IPEX_FP32_MATH_MODE

# For cosmic tagger, this improves performance:
unset IPEX_XPU_ONEDNN_LAYOUT_OPT
#setenv IPEX_XPU_ONEDNN_LAYOUT_OPT "1"

#####################################################################
# End of perf-adjustment section
#####################################################################


#####################################################################
# Environment set up, using the latest frameworks drop
#####################################################################

# Frameworks have a different oneapi backend at the moment:
module restore
# Use a newer oneapi:
# module swap oneapi/eng-compiler/2022.10.15.006 oneapi/release/2022.12.30.001

# Activate my conda install:
# source /home/cadams/miniconda3/bin/activate

# Activate my conda env:
# conda activate /home/cadams/intel-python-conda-build/envs
# module list


module load frameworks/2023.05.15.001


export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=1

# Create the command arguments:

echo "CREATE ARGS"

PYTHON_ARGUMENTS="bin/exec.py --config-name a21 framework=${FRAMEWORK} "
PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.compute_mode=XPU run.distributed=False data.data_format=${DATA_FORMAT}"
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
ZE_AFFINITY_LIST=("0.0" "1.0"  "2.0" "3.0" "4.0" "5.0" "0.1" "1.1" "2.1" "3.1" "4.1" "5.1")
CPU_AFFINITY_LIST=("0"  "16"   "32"  "52"  "68"  "84"  "8"   "24"  "40"  "60"  "76"  "92")
RUN_ID_TEMPLATE=sunspot-a21-single-tile-${FRAMEWORK}-n${NRANKS}-df${DATA_FORMAT}-p${PRECISION}-mb${LOCAL_BATCH_SIZE}-synthetic-tile-run${RUN}


for (( idx=0; idx<${NRANKS_PER_NODE}; idx++ ))
do
	echo "Hello"
	echo $idx
	GPU=${ZE_AFFINITY_LIST[$idx]}
	CPU=${CPU_AFFINITY_LIST[$idx]}
	host=$(hostname)
	run_id="${RUN_ID_TEMPLATE}/${host}/GPU${GPU}-CPU${CPU}"
	echo $run_id
	export ZE_AFFINITY_MASK=${GPU}
	numactl -C ${CPU} python $PYTHON_ARGUMENTS run.id=${run_id} output_dir=${OUTPUT_DIR_TOP}/${run_id} > /dev/null 2>&1 &
	unset ZE_AFFINITY_MASK
done

wait < <(jobs -p)

echo "DONE LOOP"

# This string is an identified to store log files:
