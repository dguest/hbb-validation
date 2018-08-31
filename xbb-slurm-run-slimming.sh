#!/usr/bin/env bash

# one of the more confusing things about slurm is that it will run
# jobs in your home directory. so the first line in the script should
# be to go to $SLURM_SUBMIT_DIR, which is set by the batch system
# before running the job.

echo "submit from $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
echo "Running on $1, building in $2"

xbb_slim_dataset.py $1 -o $2

echo "done"

