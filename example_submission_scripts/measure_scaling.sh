#!/bin/bash -l
#PBS -l select=13
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q lustre_scaling
#PBS -A Aurora_deployment

#####################################################################
# Most of the configuration is in the subscript, check there
#####################################################################

NNODES=`wc -l < $PBS_NODEFILE`

WORKDIR=/home/cadams/CosmicTagger/performance-measurement/; cd ${WORKDIR}
SUBSCRIPT=${WORKDIR}/run_performance_test.py

OUTDIR_DIR=/flare/Aurora_deployment/cadams/ct_v1_perf_output/
# OUTPUT_DIR=/lus/eagle/projects/datascience/cadams/perf-measurement-jink/

export DATE=$(date "+%F-%H:%M:%S")
echo $DATE

# 
# module use /soft/modulefiles
# module load frameworks/2023.12.15.001
# Fix missing library:
# export LD_LIBRARY_PATH=/home/rramer/test-lib:$LD_LIBRARY_PATH





# Detect the system and set up environmet and data:
if [[ $(hostname -f) == *"polaris"* ]]; 
then 
    echo "Set up for Polaris"; 
    NRANKS_PER_NODE=4

    # Set up software deps:
    module use /soft/modulefiles/
    module load conda/2024-04-29; conda activate

    # Add-ons from conda:
    source /home/cadams/Polaris/polaris_conda_2024-04-29-venv/bin/activate

    SYSTEM="polaris"

elif [[ $(hostname -f) == *"aurora"* ]];
then
    echo "Set up for Aurora";
    NRANKS_PER_NODE=12
    DATA_DIR=MISSING

    COMPUTE_MODE="XPU"
    module load frameworks/2024.1
    
    # Performance variables:
    unset ITEX_LAYOUT_OPT
    unset IPEX_XPU_ONEDNN_LAYOUT_OPT

    ITEX_FP32_MATH_MODE=TF32
    IPEX_FP32_MATH_MODE=TF32

    # This is a fix for running over 16 nodes:
    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export FI_CXI_OVFLOW_BUF_SIZE=8388608
    export FI_CXI_CQ_FILL_PERCENT=20



elif [[ $(hostname -f) == *"americas"* ]];
then
    echo "Set up for Sunspot"
    NRANKS_PER_NODE=12
    DATA_DIR=MISSING
    COMPUTE_MODE="XPU"
    
    # Performance variables:
    unset ITEX_LAYOUT_OPT
    unset IPEX_XPU_ONEDNN_LAYOUT_OPT

    ITEX_FP32_MATH_MODE=TF32
    IPEX_FP32_MATH_MODE=TF32

    # This is a fix for running over 16 nodes:
    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export FI_CXI_OVFLOW_BUF_SIZE=8388608
    export FI_CXI_CQ_FILL_PERCENT=20
    
    SYSTEM="aurora"

else
    echo "Why did Corey leave? Now my code doesn't work :'("
fi

export OMP_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8


export RUN=1

python ${SUBSCRIPT} \
-s ${SYSTEM} -i 102 \
-o ${OUTPUT_DIR}/${NNODES}nodes/run${RUN}-${DATE}/

export RUN=2

python ${SUBSCRIPT} \
-s ${SYSTEM} -i 102 \
-o ${OUTPUT_DIR}/${NNODES}nodes/run${RUN}-${DATE}/
