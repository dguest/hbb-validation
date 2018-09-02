#!/usr/bin/env bash

set -eu

COMMAND="$1"
INPUTS=${@:2}

for INPUT in ${INPUTS}; do
    N=$(jobs | egrep -i 'running' | wc -l)
    if ((N > 10)) ; then
        echo "$N jobs running, waiting"
        sleep 1
    else
        ${COMMAND} $INPUT &
    fi
done
echo "waiting for jobs to finish"
wait
