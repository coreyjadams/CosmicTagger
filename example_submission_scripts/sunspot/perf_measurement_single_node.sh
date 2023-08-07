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
OUTPUT_DIR=/lus/gila/projects/Aurora_deployment/cadams/ct-paper-measurements/
WORKDIR=/home/cadams/CosmicTagger/
cd ${WORKDIR}

CONFIG="polaris"
NRANKS_PER_NODE=12
FRAMEWORKS=("tensorflow")
# FRAMEWORKS=("torch" "tensorflow")
PRECISIONS=("float32" "float32" "bfloat16" )
MATHMODES=(FP32 TF32 FP32)

BATCH_SIZE=(1 2)


DATA_FORMAT="channels_last"
ITERATIONS=200

# Precision for CT can be float32, bfloat16, or mixed (fp16).


# ITEX_FP32_MATH_MODE=0
unset ITEX_FP32_MATH_MODE
unset IPEX_FP32_MATH_MODE
# IPEX_FP32_MATH_MODE=FP32

# For cosmic tagger, this improves performance:
# (for reference, the default is "setenv ITEX_LAYOUT_OPT \"1\" ")
unset ITEX_LAYOUT_OPT


#####################################################################
# Environment set up, using the latest frameworks drop
#####################################################################

module load frameworks/2023.05.15.001




module list


export NUMEXPR_MAX_THREADS=16


ZE_AFFINITY_LIST=("0.0" "0.1" "1.0" "1.1" "2.0" "2.1" "3.0" "3.1" "4.0" "4.1" "5.0" "5.1")
CPU_AFFINITY_LIST=( \
    "0-7,104-111" \
    "8-15,112-119" \
    "16-23,120-127" \
    "24-31,128-135" \
    "32-39,136-143" \
    "40-47,144-151" \
    "52-59,156-163" \
    "60-67,164-171" \
    "68-75,172-179" \
    "76-83,180-187" \
    "84-91,188-195" \
    "92-99,196-203" \
)
export CPU_AFFINITY="verbose,list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203"

#####################################################################
# End of environment setup section
#####################################################################

#####################################################################
# JOB LAUNCH
# Note that this example targets a SINGLE TILE
#####################################################################
echo $FRAMEWORKS
for framework in ${FRAMEWORKS[@]};
do
    echo "Framework: ${framework}"
    for batch_size in ${BATCH_SIZE[@]};
    do
        echo "Batch size: ${batch_size}"
        for (( i_prec=0; i_prec<${#PRECISIONS[*]}; ++i_prec));
        do
            # Set the precision in both frameworks
            export ITEX_FP32_MATH_MODE="${MATHMODES[$i_prec]}"
            export IPEX_FP32_MATH_MODE="${MATHMODES[$i_prec]}"
            precision="${PRECISIONS[$i_prec]}"
            math_mode="${MATHMODES[$i_prec]}"
            precisionStr="${precision}-${math_mode}"
            echo $"Precision: ${precision}"
            #configure the run id:
            
            # Configure the python args:
            PYTHON_ARGUMENTS="${WORKDIR}/bin/exec.py --config-name ${CONFIG} "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} framework=${framework} "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.compute_mode=XPU "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.distributed=False "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} data=synthetic "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} data.data_format=${DATA_FORMAT} "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.precision=${precision} "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.minibatch_size=${batch_size} "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.iterations=${ITERATIONS}"
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} mode.checkpoint_iteration=20000"

            RUN_ID_TEMPLATE=sunspot-${CONFIG}-${framework}-df${DATA_FORMAT}-p${precisionStr}${math_mode}-mb${batch_size}

            for (( idx=0; idx<${NRANKS_PER_NODE}; idx++ ));
            do
                GPU=${ZE_AFFINITY_LIST[$idx]}
                CPU=${CPU_AFFINITY_LIST[$idx]}
                host=$(hostname)
                run_id="${RUN_ID_TEMPLATE}/${host}/GPU${GPU}"
                echo $run_id
                export ZE_AFFINITY_MASK=${GPU}
                echo ${CPU}
                # numactl -C ${CPU} python $PYTHON_ARGUMENTS run.id=${run_id} output_dir=${OUTPUT_DIR}/${run_id} 2>&1 &
                # numactl -C ${CPU} python $PYTHON_ARGUMENTS run.id=${run_id} output_dir=${OUTPUT_DIR}/${run_id} > /dev/null 2>&1 &
                unset ZE_AFFINITY_MASK
            done
            jobs -p

            # Wait for each gpu's job to complete:
            wait $(jobs -p)


            # Run the multitile version:
            run_id="${RUN_ID_TEMPLATE}/${host}/fullnode/"
            mpiexec -n ${NRANKS_PER_NODE} -ppn ${NRANKS_PER_NODE} \
            --cpu-bind ${CPU_AFFINITY} \
            python bin/exec.py \
            --config-name ${CONFIG} \
            framework=${framework} \
            output_dir=${OUTPUT_DIR}/${run_id} \
            run.id=${run_id} \
            run.compute_mode=XPU \
            run.distributed=True \
            data=synthetic \
            data.data_format=${DATA_FORMAT} \
            run.precision=${precision} \
            run.minibatch_size=${batch_size} \
            run.iterations=$ITERATIONS
        done
    done
done



#####################################################################
# Launch the script
# This section is to outline what the command is doing
#
# python bin/exec.py \						# Script entrqy point
# --config-name a21 \						# Aurora acceptance model
# framework=tensorflow \					# TF is default, but explicitly setting it
# output_dir=${OUTPUT_DIR}/${run_id} \		# Direct the output to this folder
# run.id=${run_id} \						# Pass the unique runID
# run.compute_mode=XPU \					# Explicitly set XPU as the target accelerator
# run.distributed=False \					# SINGLE-TILE: disable all collectives
# data.data_format=${DATA_FORMAT} \			# Set data format per user spec
# run.precision=${PRECISION} \				# Set precision per user spec
# run.minibatch_size=${LOCAL_BATCH_SIZE} \	# Set minibatch size per user spec
# run.iterations=250						# Run for 250 iterations.
#####################################################################

# # export CCL_LOG_LEVEL="WARN"

# ulimit -c 0

# mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} \
# --cpu-bind ${CPU_AFFINITY} \
# python bin/exec.py \
# --config-name polaris \
# framework=torch \
# output_dir=${OUTPUT_DIR}/${run_id} \
# run.id=${run_id} \
# run.compute_mode=XPU \
# run.distributed=False \
# data=synthetic \
# data.data_format=${DATA_FORMAT} \
# run.precision=${PRECISION} \
# run.minibatch_size=${GLOBAL_BATCH_SIZE} \
# run.iterations=500
