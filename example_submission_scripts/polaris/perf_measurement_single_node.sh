#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q preemptable
#PBS -A datascience
#PBS -l filesystems=home:grand


#####################################################################
# These are my own personal directories,
# you will need to change these.
#####################################################################
OUTPUT_DIR=/lus/grand/projects/datascience/cadams/ct-paper-measurements-2.13/
WORKDIR=/home/cadams/Polaris/CosmicTagger/
cd ${WORKDIR}

CONFIG="a21"
NRANKS_PER_NODE=4
# FRAMEWORKS=("torch" "tensorflow" "tensorflow")
# XLA_FLAGS=("" "" "--tf_xla_auto_jit=2")
FRAMEWORKS=("tensorflow")
XLA_FLAGS=("")
PRECISIONS=("float32" "float32" "mixed" )
MATHMODES=(0 "" "")

# BATCH_SIZE=(1 2)
BATCH_SIZE=(8 12)


DATA_FORMAT="channels_last"
ITERATIONS=200

# Precision for CT can be float32, bfloat16, or mixed (fp16).


#####################################################################
# Environment set up, using the latest frameworks drop
#####################################################################


module load conda
# conda activate
# conda activate torch2.X
conda activate tf2.13

export NUMEXPR_MAX_THREADS=16


GPU_AFFINITY_LIST=("0" "1" "2" "3")
CPU_AFFINITY_LIST=( \
    "24-31,56-63" \
    "16-23,48-55" \
    "8-15,40-47" \
    "0-7,32-39" \
)
export CPU_AFFINITY="numa"

#####################################################################
# End of environment setup section
#####################################################################

#####################################################################
# JOB LAUNCH
# Note that this example targets a SINGLE TILE
#####################################################################
echo $FRAMEWORKS
f_counter=1
for framework in ${FRAMEWORKS[@]};
do
    unset TF_XLA_FLAGS
    xla_val="${XLA_FLAGS[$f_counter]}"
    export TF_XLA_FLAGS=${xla_val}
    echo "XLA flags: ${TF_XLA_FLAGS}"
    let f_counter=${f_counter}+1
    echo "Framework: ${framework}-${f_counter}"
    echo "f_counter: ${f_counter}"
    for batch_size in ${BATCH_SIZE[@]};
    do
        echo "Batch size: ${batch_size}"
        for (( i_prec=0; i_prec<${#PRECISIONS[*]}; ++i_prec));
        do
            # Set the precision in both frameworks
            export NVIDIA_TF32_OVERRIDE="${MATHMODES[$i_prec]}"
            precision="${PRECISIONS[$i_prec]}"
            math_mode="${MATHMODES[$i_prec]}"
            precisionStr="${precision}-${math_mode}"
            echo $"Precision: ${precision}"
            #configure the run id:
            
            # Configure the python args:
            PYTHON_ARGUMENTS="${WORKDIR}/bin/exec.py --config-name ${CONFIG} "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} framework=${framework} "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.compute_mode=GPU "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.distributed=False "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} data=synthetic "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} data.data_format=${DATA_FORMAT} "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.precision=${precision} "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.minibatch_size=${batch_size} "
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} run.iterations=${ITERATIONS}"
            PYTHON_ARGUMENTS="${PYTHON_ARGUMENTS} mode.checkpoint_iteration=20000"

            RUN_ID_TEMPLATE=polaris-${CONFIG}-${framework}-${f_counter}-df${DATA_FORMAT}-p${precisionStr}-mb${batch_size}

            for (( idx=0; idx<${NRANKS_PER_NODE}; idx++ ));
            do
                GPU=${GPU_AFFINITY_LIST[$idx]}
                CPU=${CPU_AFFINITY_LIST[$idx]}
                host=$(hostname)
                run_id="${RUN_ID_TEMPLATE}/${host}/GPU${GPU}"
                echo $run_id
                export CUDA_VISIBLE_DEVICES=${GPU}
                echo ${CPU}
                numactl -C ${CPU} python $PYTHON_ARGUMENTS run.id=${run_id} output_dir=${OUTPUT_DIR}/${run_id}  2>&1 &
                # numactl -C ${CPU} python $PYTHON_ARGUMENTS run.id=${run_id} output_dir=${OUTPUT_DIR}/${run_id} > /dev/null 2>&1 &
                unset CUDA_VISIBLE_DEVICES
            done
            jobs -p
			
            # Wait for each gpu's job to complete:
            wait $(jobs -p)
		    exit -1

            # Run the multitile version:
            run_id="${RUN_ID_TEMPLATE}/${host}/fullnode/"
            mpiexec -n ${NRANKS_PER_NODE} -ppn ${NRANKS_PER_NODE} \
            --cpu-bind ${CPU_AFFINITY} \
            python bin/exec.py \
            --config-name ${CONFIG} \
            framework=${framework} \
            output_dir=${OUTPUT_DIR}/${run_id} \
            run.id=${run_id} \
            run.compute_mode=GPU \
            run.distributed=True \
            data=synthetic \
            data.data_format=${DATA_FORMAT} \
            run.precision=${precision} \
            run.minibatch_size=${batch_size} \
            run.iterations=$ITERATIONS
        done
    done
done


