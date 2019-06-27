#!/bin/sh
#COBALT -t 60
#COBALT -n 8
#COBALT -q debug-cache-quad
#COBALT -A datascience
#COBALT --attrs nox11

export N_NODES=8
let BATCH_SIZE=1*${N_NODES}

# Set up software deps:
source /projects/datascience/cadams/software/miniconda/bin/activate
export LD_LIBRARY_PATH=/projects/datascience/cadams/software/miniconda/lib/:$LD_LIBRARY_PATH

WORKDIR=/home/cadams/Theta/DLP3/CosmicTagger/

export KMP_AFFINITY=granularity=fine,verbose,compact 
export OMP_NUM_THREADS=64
export KMP_BLOCKTIME=0
export MKLDNN_VERBOSE=0 
export MPICH_MAX_THREAD_SAFETY=multiple

echo "This is the conda installation"

aprun -n ${N_NODES} -N 1 \
-cc depth \
-j 1 \
-d 64 \
python ${WORKDIR}/bin/exec.py train -d \
-f /projects/datascience/cadams/datasets/SBND/H5/cosmic_tagging_downsample/cosmic_tagging_downsample_train_sparse.h5 \
-mb ${BATCH_SIZE} \
-m CPU \
-ld ${WORKDIR}/log/tf/${NODES}nodes_test_conda_hvd/ \
-i 100   \
--n-initial-filters 12 \
--network-depth 5 \
--checkpoint-iteration 500 \
--optimizer adam \
-lr 0.0001 \
--balance-loss False \
--batch-norm False 

# --aux-file /projects/datascience/cadams/datasets/SBND/H5/cosmic_tagging_downsample/cosmic_tagging_downsample_test_sparse.h5 \
# --aux-minibatch-size ${BATCH_SIZE}

