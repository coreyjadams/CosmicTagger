#!/bin/bash -l
#PBS -l select=1:system=sunspot
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q workq
#PBS -A Aurora_deployment

#####################################################################
# Most of the configuration is in the subscript, check there
#####################################################################

NNODES=`wc -l < $PBS_NODEFILE`

WORKDIR=/home/cadams/CosmicTagger/performance-measurement/; cd ${WORKDIR}
SUBSCRIPT=${WORKDIR}/run_performance_test.py

export DATE=$(date "+%F-%H:%M:%S")
echo $DATE

module load frameworks/2023.05.15.001

export RUN=1

python ${SUBSCRIPT} \
-s sunspot \
-o /lus/gila/projects/Aurora_deployment/cadams/performance-tests-${NNODES}nodes/run${RUN}-${DATE}/

export RUN=2

python ${SUBSCRIPT} \
-s sunspot \
-o /lus/gila/projects/Aurora_deployment/cadams/performance-tests-${NNODES}nodes/run${RUN}-${DATE}/
