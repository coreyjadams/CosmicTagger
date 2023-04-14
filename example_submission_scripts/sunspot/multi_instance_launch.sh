#!/bin/bash -l
#PBS -l select=42:system=sunspot
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q workq
#PBS -A Aurora_deployment

#####################################################################
# Most of the configuration is in the subscript, check there
#####################################################################

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
let NRANKS=${NNODES}*${NRANKS_PER_NODE}

SUBSCRIPT=/home/cadams/CosmicTagger/example_submission_scripts/sunspot/single_node_instance_spawn.sh

export DATE=$(date +%F-%I)

export FRAMEWORK="tensorflow"

export RUN=1
mpirun -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT

export RUN=2
mpirun -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT

export RUN=3
mpirun -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT

export FRAMEWORK="torch"

export RUN=1
mpirun -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT

export RUN=2
mpirun -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT

export RUN=3
mpirun -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT
