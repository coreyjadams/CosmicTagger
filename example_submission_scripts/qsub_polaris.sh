#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience


DIR=/home/cadams/Polaris/CosmicTagger/example_submission_scripts/

export MODEL=a21
export FRAMEWORK=torch
export ITERATIONS=5000

$DIR/run_ct.sh
