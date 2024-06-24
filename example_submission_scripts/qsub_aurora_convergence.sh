#!/bin/bash -l
#PBS -l select=128
#PBS -l place=scatter
#PBS -l walltime=2:30:00
#PBS -q lustre_scaling
#PBS -l filesystems=home:eagle
#PBS -A Aurora_deployment



export WORK_DIR=/home/cadams/CosmicTagger-latest/
export OUTPUT_DIR=/flare/Aurora_deployment/cadams/ct-output-convergence/
export MODEL=uresnet2
export FRAMEWORK=torch
export PRECISION=bfloat16
export DISTRIBUTED_MODE=DDP
export DOWNSAMPLE=0
export EPOCHS=150
export LOCAL_BATCH_SIZE=1

export CT_OVERRIDES="network.depthwise=True network.kernel_size=3"

$WORK_DIR/example_submission_scripts/run_ct.sh
