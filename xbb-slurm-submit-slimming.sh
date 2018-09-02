#!/usr/bin/env bash

set -eu

_usage() {
    echo "usage: ${0##*/} [-hf] -i <inputs> [-o outputs] [-l logs]"
}

OUTPUTS=slimmed
INPUTS=''
LOGS=logs

_help() {
    _usage
    cat <<EOF

Submit some batch jobs to slim things

Options:
 -h: get help
 -f: delete existing files
 -i <input_dir>: directory where the input datasets live
 -o <output_dir>: where the outputs will go, default "${OUTPUTS}"
 -l <logs_dir>: where to put logs, default "${LOGS}"

EOF
}

FORCE=''

get_options() {
    local opt
    while getopts ":hfi:o:" opt $@; do
	      case $opt in
	          h) _help; return 1;;
            f) FORCE=TRUE ;;
	          i) INPUTS=${OPTARG} ;;
            o) OUTPUTS=${OPTARG} ;;
            l) LOGS=${OPTARG} ;;
	          # handle errors
	          \?) _usage; echo "Unknown option: -$OPTARG" >&2; exit 1;;
            :) _usage; echo "Missing argument for -$OPTARG" >&2; exit 1;;
            *) _usage; echo "Unimplemented option: -$OPTARG" >&2; exit 1;;
	      esac
    done
    shift $(($OPTIND - 1))

    if [[ -z $INPUTS || ! -d ${INPUTS} ]] ; then
        echo "Missing inputs dir" >&2
        exit 1
    fi

}

# __________________________________________________________________
# run all the things

# get options
get_options $@

if [[ -d ${OUTPUTS} ]]; then
    if [[ $FORCE ]]; then
        echo "removing ${OUTPUTS}"
        rm -r ${OUTPUTS}
    else
        echo "found ${OUTPUTS}, refusing to overwrite" >&2
        exit 1
    fi
fi

# make output log directory
mkdir -p ${LOGS}

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
BOPTS="-t 60 -p atlas_all -o ${LOGS}/out-%j.txt -e ${LOGS}/error-%j.txt"

# Actually submit the job
#
for DS in $INPUTS/*
do
    if [[ ! -d $DS ]] ; then
        echo "skipping $DS, it is not a directory!" >&2
        continue
    fi
    sbatch ${BOPTS} xbb-slurm-run-slimming.sh $DS $OUTPUTS
done
