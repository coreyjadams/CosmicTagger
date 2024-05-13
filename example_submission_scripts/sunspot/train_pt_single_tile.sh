#!/bin/bash -l
#PBS -l select=1:system=sunspot
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q workq
#PBS -A Aurora_deployment


#####################################################################
# These are my own personal directories,
# you will need to change these.
#####################################################################
OUTPUT_DIR=/lus/gila/projects/Aurora_deployment/cadams/ct_output/
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
LOCAL_BATCH_SIZE=8

# NOTE: batch size 8 works ok, batch size 16 core dumps, haven't explored
# much in between.  reduced precision should improve memory usage.

#####################################################################
# FRAMEWORK Variables that make a performance difference for tf:
#####################################################################

# Toggle tf32 on (or don't):
# IPEX_FP32_MATH_MODE=0 # FP32
# IPEX_FP32_MATH_MODE=1 # TF32
# IPEX_FP32_MATH_MODE=2 # BF32 ??

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

module restore

module use /soft/preview-modulefiles/24.086.0

module load frameworks/2024.04.15.002

# This removes the version frameworks automatically loads:
module unload graphics-compute-runtime

# This is the intel_compute_runtime equivalent as best I can tell:
# module load intel_compute_runtime/release/803.29

# This is the one targeted for next release
module load intel_compute_runtime/release/821.36

# Dump the loaded modules:
module list

export NUMEXPR_MAX_THREADS=8
export OMP_NUM_THREADS=8


#####################################################################
# End of environment setup section
#####################################################################

#####################################################################
# JOB LAUNCH
# Note that this example targets a SINGLE TILE
#####################################################################


# This string is an identified to store log files:
run_id=agama-test

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

ulimit -c 0

# Launch the script
python bin/exec.py \
--config-name a21 \
framework=torch \
output_dir=${OUTPUT_DIR}/${run_id} \
run.id=${run_id} \
run.compute_mode=XPU \
run.distributed=False \
data.data_format=${DATA_FORMAT} \
run.precision=${PRECISION} \
run.minibatch_size=${LOCAL_BATCH_SIZE} \
run.iterations=100
