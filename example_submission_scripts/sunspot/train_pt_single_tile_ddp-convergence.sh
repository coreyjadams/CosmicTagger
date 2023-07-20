#!/bin/bash -l
#PBS -l select=16:system=sunspot
#PBS -l place=scatter
#PBS -l walltime=0:20:00
#PBS -q workq
#PBS -A Aurora_deployment


#####################################################################
# These are my own personal directories,
# you will need to change these.
#####################################################################
OUTPUT_DIR=/lus/gila/projects/Aurora_deployment/cadams/ct_output-master/
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
LOCAL_BATCH_SIZE=2
let BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

# NOTE: batch size 8 works ok, batch size 16 core dumps, haven't explored
# much in between.  reduced precision should improve memory usage.

#####################################################################
# FRAMEWORK Variables that make a performance difference for tf:
#####################################################################

# Toggle tf32 on (or don't):
# IPEX_FP32_MATH_MODE=TF32
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
module list

frameworks="2023.05.15"

# module load frameworks/${frameworks}.001
module load frameworks/2023.05.15.001
source /home/cadams/frameworks-2023-05-15-extension/bin/activate
# source /home/cadams/frameworks-${frameworks}-extension/bin/activate

export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=1

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


# This string is an identified to store log files:
run_id=${frameworks}-ddp-n${NRANKS}


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

# Launch the script
mpiexec -np ${NRANKS} -ppn ${NRANKS_PER_NODE} \
--cpu-bind ${CPU_AFFINITY} \
python bin/exec.py \
framework=torch \
output_dir=${OUTPUT_DIR}/${run_id} \
run.id=${run_id} \
run.compute_mode=XPU \
run.distributed=True \
data.data_directory=/lus/gila/projects/Aurora_deployment/cadams/cosmic_tagger/ \
data.data_format=${DATA_FORMAT} \
network.n_initial_filters=32 \
run.precision=${PRECISION} \
run.minibatch_size=${BATCH_SIZE} \
run.iterations=25000
