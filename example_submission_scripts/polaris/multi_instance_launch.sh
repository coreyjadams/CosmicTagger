#!/bin/bash -l
#PBS -l select=256:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=home:grand

#####################################################################
# Most of the configuration is in the subscript, check there
#####################################################################

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
let NRANKS=${NNODES}*${NRANKS_PER_NODE}

SUBSCRIPT=/home/cadams/Polaris/CosmicTagger/example_submission_scripts/polaris/single_node_instance_spawn.sh

export DATE=$(date +%F-%I)

export FRAMEWORK="tensorflow"

export RUN=1
mpiexec -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT

export RUN=2
mpiexec -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT

export RUN=3
mpiexec -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT

export FRAMEWORK="torch"

export RUN=1
mpiexec -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT

export RUN=2
mpiexec -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT

export RUN=3
mpiexec -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT
