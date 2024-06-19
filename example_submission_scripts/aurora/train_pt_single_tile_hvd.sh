#!/bin/bash -l
#PBS -l select=4
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q M242798
#PBS -A Aurora_deployment


#####################################################################
# These are my own personal directories,
# you will need to change these.
#####################################################################
OUTPUT_DIR=/home/cadams/ct_output/
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

# This is a fix for running over 16 nodes:
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OVFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20

export FI_LOG_LEVEL=warn
export FI_LOG_PROV=tcp



#####################################################################
# APPLICATION Variables that make a performance difference for tf:
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
LOCAL_BATCH_SIZE=8

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

module use /soft/modulefiles
module load frameworks/.2023.08.15.002

# Fix for EVP_* symbols issue:
# export LD_LIBRARY_PATH=/home/rramer/test-lib:$LD_LIBRARY_PATH


export NUMEXPR_MAX_THREADS=1


#####################################################################
# End of environment setup section
#####################################################################

#####################################################################
# JOB LAUNCH
# Note that this example targets a SINGLE TILE
#####################################################################


# This string is an identified to store log files:
run_id=aurora-a21-single-tile-hvd-n${NRANKS}-df${DATA_FORMAT}-p${PRECISION}-mb${LOCAL_BATCH_SIZE}


#####################################################################
# Launch the script
# This section is to outline what the command is doing
#
# python bin/exec.py \						# Script entry point
# --config-name a21 \						# Aurora acceptance model
# framework=torch \							# Switch to torch here
# output_dir=${OUTPUT_DIR}/${run_id} \		# Direct the output to this folder
# run.id=${run_id} \						# Pass the unique runID
# run.compute_mode=XPU \					# Explicitly set XPU as the target accelerator
# run.distributed=True \					# Use collectives now
# data.data_format=${DATA_FORMAT} \			# Set data format per user spec
# run.precision=${PRECISION} \				# Set precision per user spec
# run.minibatch_size=${LOCAL_BATCH_SIZE} \	# Set minibatch size per user spec
# run.iterations=250						# Run for 250 iterations.
#####################################################################

export CCL_LOG_LEVEL="WARN"
export CPU_AFFINITY="verbose,list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203"

ulimit -c 0

echo "About to mpiexec"
date

# Launch the script
mpiexec -np ${NRANKS} -ppn ${NRANKS_PER_NODE} \
--cpu-bind ${CPU_AFFINITY} \
python bin/exec.py \
--config-name a21 \
framework=torch \
output_dir=${OUTPUT_DIR}/${run_id} \
run.id=${run_id} \
run.compute_mode=XPU \
run.distributed=True \
framework.distributed_mode=horovod \
data.data_format=${DATA_FORMAT} \
run.precision=${PRECISION} \
run.minibatch_size=${LOCAL_BATCH_SIZE} \
run.iterations=500
