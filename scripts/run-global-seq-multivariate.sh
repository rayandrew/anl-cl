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

# Initialize environment
source $INIT_SCRIPT

# Run experiment here
# MACHINE_ID="m_25"
# Y_VAR="disk"
# STRATEGY="gdumb"
EXP_OUT_DIR="$EXP_DIR/out/global/seq/multivariate"
SEQ_LEN=5
EVAL_ONLY=0

# DATA PREPROCESSING OPTIONS
WINDOW_SIZE=75
THRESHOLD=300 # for voting
PREPROCESS_DATA_DIR="$EXP_DIR/preprocessed_data"

mkdir -p $PREPROCESS_DATA_DIR
mkdir -p $EXP_OUT_DIR

run() {
    local machine_id=$1
    local y_var=$2
    local strategy=$3
    local seq_len=$4
    local eval_only=$5

    local data_file="$PREPROCESS_DATA_DIR/$machine_id/${machine_id}_${WINDOW_SIZE}-${THRESHOLD}/${machine_id}_${y_var}.csv"
    if [[ ! -f $data_file ]]; then
        echo ">>> Preprocessing data for machine_id=$machine_id, y_var=$y_var"
        echo
        # python drift_detection.py data/mu/m_25.csv -y cpu --window_size 75 --threshold 300
        python drift_detection.py "$ALIBABA_MU/$machine_id.csv" \
            -o "$PREPROCESS_DATA_DIR" \
            -y $y_var \
            --window_size $WINDOW_SIZE \
            --threshold $THRESHOLD
        echo
    else
        echo ">>> Data for machine_id=$machine_id, y_var=$y_var, window_size=$WINDOW_SIZE, and threshold=$THRESHOLD already exists, skipping preprocessing data"
        echo
    fi
    
    if [[ $eval_only == 0 && ! -f "$EXP_OUT_DIR/${machine_id}_${y_var}/${strategy}/done" ]]; then
        echo ">>> Train SEQUENTIAL MULTIVARIATE model machine_id=$machine_id, y_var=$y_var, strategy=$strategy"
        echo
        python main.py \
            -f "$EXP_DIR/preprocessed_data/$machine_id/${machine_id}_${WINDOW_SIZE}-${THRESHOLD}/${machine_id}_${y_var}.csv" \
            -m "${machine_id}_${strategy}" \
            -s $strategy \
            -o "$EXP_OUT_DIR/${machine_id}_${y_var}/${strategy}/" \
            -y $y_var \
            --seq --seq_len $seq_len
        touch "$EXP_OUT_DIR/${machine_id}_${y_var}/${strategy}/done"
        echo
    else
        echo ">>> SEQUENTIAL MULTIVARIATE model for machine_id=$machine_id, y_var=$y_var, strategy=$strategy already exists, skipping training"
        echo 
    fi

    echo ">>> Evaluate SEQUENTIAL MULTIVARIATE model machine_id=$machine_id, y_var=$y_var, strategy=$strategy"  
    echo
    python eval.py \
        -f "$EXP_DIR/preprocessed_data/$machine_id/${machine_id}_${WINDOW_SIZE}-${THRESHOLD}/${machine_id}_${y_var}.csv" \
        -o "$EXP_OUT_DIR/${machine_id}_${y_var}/${strategy}" \
        -m "$EXP_OUT_DIR/${machine_id}_${y_var}/${strategy}/${machine_id}_${strategy}.pt" \
        -y $y_var \
        --plot \
        --seq --seq_len $seq_len
}

# run "m_25" "cpu" "ewc" $SEQ_LEN $EVAL_ONLY

# STRATEGIES=("naive" "ewc" "gss" "lwf" "agem" "gdumb")
STRATEGIES=("naive" "ewc" "gss")

## for machine_id in $(ls $ALIBABA_MU); do
for strategy in "${STRATEGIES[@]}"; do
    for machine_id in "m_25" "m_881"; do
        for y_var in "cpu" "mem" "disk"; do
            echo ">>> Running SEQUENTIAL MULTIVARIATE pipeline for machine_id=$machine_id, y_var=$y_var, strategy=$strategy"
            run $machine_id $y_var $strategy $SEQ_LEN $EVAL_ONLY
        done
    done
done