#!/usr/bin/env bash

set -e

machine_id=$1
model=$2
shift
shift

if [ -z "$model" ]; then
    echo "Usage: $0 <machine_id> <model>"
    exit 1
fi

run() {
    machine_id=$1
    y_var=$2
    model=$3
    n_exp=$4

    echo "Train machine_id=$machine_id, y_var=$y_var, model=$model, n_exp=$n_exp"
    echo
    python main.py -f "./data/$machine_id.csv" -nl 10 -x "$n_exp" -m "$machine_id"_"$model" -o "out/$machine_id/$model" -y "$y_var" -s "$model"
    echo

    echo "Evaluate machine_id=$machine_id, y_var=$y_var, model=$model"  
    echo
    python eval.py -f "./data/$machine_id.csv" -nl 10 -m "./out/$machine_id/$model/$machine_id"_"$model.pt" -o "out/$machine_id/$model" -p -y "$y_var"
}

if [[ "$machine_id" == "m_25" ]]; then
    run $machine_id "disk" $model 2
    # run $machine_id "mem_util_percent" $model 2
elif [[ "$machine_id" == "m_881" ]]; then
    run $machine_id "mem" $model 3
else
    echo "Unknown machine id: $machine_id"
    exit 1
fi
