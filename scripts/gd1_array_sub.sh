#!/bin/bash
#SBATCH -A m4474
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00:00
#SBATCH --array=0-20
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
 
PATCH=$SLURM_ARRAY_TASK_ID
 
source ../.venv/bin/activate
python fullgd1splitgpus_dec.py --patch "$PATCH"