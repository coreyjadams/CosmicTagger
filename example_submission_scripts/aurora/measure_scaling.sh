#!/bin/bash -l
#PBS -l select=128
#PBS -l place=scatter
#PBS -l walltime=1:30:00
#PBS -q M243203
#PBS -A Aurora_deployment

#####################################################################
# Most of the configuration is in the subscript, check there
#####################################################################

NNODES=`wc -l < $PBS_NODEFILE`

WORKDIR=/home/cadams/CosmicTagger/performance-measurement/; cd ${WORKDIR}
SUBSCRIPT=${WORKDIR}/run_performance_test.py

export DATE=$(date "+%F-%H:%M:%S")
echo $DATE

module use /soft/modulefiles
module load frameworks/.2023.08.15.002 
# Fix missing library:
# export LD_LIBRARY_PATH=/home/rramer/test-lib:$LD_LIBRARY_PATH

# Performance variables:
unset ITEX_LAYOUT_OPT
unset IPEX_XPU_ONEDNN_LAYOUT_OPT

ITEX_FP32_MATH_MODE=TF32
IPEX_FP32_MATH_MODE=TF32

# This is a fix for running over 16 nodes:
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OVFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20



export RUN=1

python ${SUBSCRIPT} \
-s aurora -i 102 \
-o /home/cadams/performance-tests/${NNODES}nodes/run${RUN}-${DATE}/

export RUN=2

python ${SUBSCRIPT} \
-s aurora -i 102 \
-o /home/cadams/performance-tests/${NNODES}nodes/run${RUN}-${DATE}/
