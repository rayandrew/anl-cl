#!/bin/bash
#SBATCH --job-name=acl    
#SBATCH --account=FASTBAYES
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             
#SBATCH --time=24:00:00          
#SBATCH --output=./stdout/stdout_run_acl.%j
#SBATCH --output=./stdout/stdout_run_acl.%j

set -e

# User Configuration
EXP_DIR=$PWD
INIT_SCRIPT=$PWD/scripts/1-init-environment.sh
SLURM_CPUS_PER_TASK=5
SLURM_GPUS_PER_TASK=1
ALIBABA_MU="/lcrc/project/FastBayes/rayandrew/machine_usage"
ALIBABA_BI="/lcrc/project/FastBayes/rayandrew/data-sorted-by-ins-start"

# Initialize environment
source $INIT_SCRIPT

# Run experiment here
# MACHINE_ID="m_25"
# Y_VAR="disk"
# STRATEGY="gdumb"
EXP_OUT_DIR="$EXP_DIR/out/local/non-seq"
EVAL_ONLY=0

# DATA PREPROCESSING OPTIONS
WINDOW_SIZE=75
THRESHOLD=300 # for voting
PREPROCESS_DATA_DIR="$EXP_DIR/preprocessed_data/local"
GROUPED_LOCAL_DATA_DIR="$EXP_DIR/preprocessed_local_data"

mkdir -p $PREPROCESS_DATA_DIR
mkdir -p $EXP_OUT_DIR

run() {
    local machine_id=$1
    local y_var=$2
    local strategy=$3
    local eval_only=$4

    local grouped_data_file="$GROUPED_LOCAL_DATA_DIR/${machine_id}.csv"
    local data_file="$PREPROCESS_DATA_DIR/local/$machine_id/${machine_id}_${WINDOW_SIZE}-${THRESHOLD}/${machine_id}_${y_var}.csv"

    if [[ ! -f $grouped_data_file ]]; then
        echo ">>> Grouping local data for machine_id=$machine_id"
        echo

        python prepare_local_data.py \
            --mu-data "$ALIBABA_MU/${machine_id}.csv" \
            --bi-data "$ALIBABA_BI/${machine_id}.csv" \
            --output "$GROUPED_LOCAL_DATA_DIR"
    else
        echo ">>> Grouped local data for machine_id=$machine_id already exists, skipping grouping local data"
        echo
    fi

    if [[ ! -f $data_file ]]; then
        echo ">>> Preprocessing data for machine_id=$machine_id, y_var=$y_var"
        echo

        python drift_detection.py "$GROUPED_LOCAL_DATA_DIR/$machine_id.csv" \
            -o "$PREPROCESS_DATA_DIR" \
            -y $y_var \
            --window_size $WINDOW_SIZE \
            --threshold $THRESHOLD \
            --local
        echo
    else
        echo ">>> [LOCAL] Data for machine_id=$machine_id, y_var=$y_var, window_size=$WINDOW_SIZE, and threshold=$THRESHOLD already exists, skipping preprocessing data"
        echo
    fi
    
    if [[ $eval_only == 0 && ! -f "$EXP_OUT_DIR/${machine_id}_${y_var}/${strategy}/done" ]]; then
        echo ">>> [LOCAL] Train non-seq model for machine_id=$machine_id, y_var=$y_var, strategy=$strategy"
        echo
        python main.py \
            -f "$data_file" \
            -m "${machine_id}_${strategy}" \
            -s $strategy \
            -o "$EXP_OUT_DIR/${machine_id}_${y_var}/${strategy}/" \
            -y $y_var \
            --local
        touch "$EXP_OUT_DIR/${machine_id}_${y_var}/${strategy}/done"
        echo
    else
        echo ">>> [LOCAL] non-seq model for machine_id=$machine_id, y_var=$y_var, strategy=$strategy already exists, skipping training"
        echo 
    fi

    echo ">>> [LOCAL] Evaluate non-seq model machine_id=$machine_id, y_var=$y_var, strategy=$strategy"  
    echo
    python eval.py \
        -f "$data_file" \
        -o "$EXP_OUT_DIR/${machine_id}_${y_var}/${strategy}" \
        -m "$EXP_OUT_DIR/${machine_id}_${y_var}/${strategy}/${machine_id}_${strategy}.pt" \
        -y $y_var \
        --plot \
        --local
}

# run "m_25" "cpu" "ewc" $EVAL_ONLY

STRATEGIES=("naive" "ewc" "gss" "lwf" "agem" "gdumb")

# for machine_id in $(ls $ALIBABA_MU); do
for machine_id in "m_25" "m_881"; do
    for y_var in "cpu" "mem" "disk"; do
        for strategy in "${STRATEGIES[@]}"; do
            echo ">>> Running machine_id=$machine_id, y_var=$y_var, strategy=$strategy"
            run $machine_id $y_var $strategy $EVAL_ONLY
        done
    done
done
