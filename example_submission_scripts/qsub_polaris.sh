#!/bin/bash -l
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l walltime=2:30:00
#PBS -q prod
#PBS -l filesystems=home:eagle
#PBS -A datascience



export WORK_DIR=/home/cadams/Polaris/CosmicTagger2/
export OUTPUT_DIR=/lus/eagle/projects/datascience/cadams/junk-ct2-output/
export MODEL=uresnet2
export FRAMEWORK=torch
export PRECISION=float32
export DISTRIBUTED_MODE=DDP
export DOWNSAMPLE=1
export EPOCHS=100
export LOCAL_BATCH_SIZE=1

$WORK_DIR/example_submission_scripts/run_ct.sh
