#!/usr/bin/env bash

set -eu

# make output log directory
mkdir -p logs

# Set the options. You can get a lot more information bo looking at
# the help! Just look at "sbatch -h".
#
# You have to use '-p atlas_all' to use our queue.
#
# You might need to play with -c if you're expecting the job to spawn
# multiple processes.
#
# The '-o' and '-e' are to specify where the outputs go. Note that the
# time limit, '-t' is in minutes.
#
BOPTS="-t 60 -p atlas_all -o logs/out-%j.txt -e logs/error-%j.txt"

# Actually submit the job
#
for DS in $1/*
do
    sbatch ${BOPTS} xbb-slurm-run-slimming.sh $DS $2
done
