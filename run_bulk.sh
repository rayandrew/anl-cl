#!/usr/bin/env bash

set -e

machine_id=$1
shift

if [ -z "$machine_id" ]; then
    echo "Usage: $0 <machine_id>"
    exit 1
fi

STRATEGIES=("naive" "ewc" "gss" "lwf" "agem" "gdumb")

for strategy in "${STRATEGIES[@]}"; do
    ./run.sh $machine_id $strategy
done

