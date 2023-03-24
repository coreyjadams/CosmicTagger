#!/bin/bash -l
#PBS -l select=82:system=sunspot
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q workq
#PBS -A Aurora_deployment

#####################################################################
# Most of the configuration is in the subscript, check there
#####################################################################

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
let NRANKS=${NNODES}*${NRANKS_PER_NODE}

SUBSCRIPT=/home/cadams/CosmicTagger/single_node_instance_spawn.sh

mpirun -n ${NRANKS} -ppn 1 --cpu-bind=none $SUBSCRIPT
