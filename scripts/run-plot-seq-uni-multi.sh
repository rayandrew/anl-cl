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

LOCAL=0
SEQ_LEN=5
N_LABELS=10

data_id="global"
if [[ $LOCAL == 1 ]]; then
    data_id="local"
fi

run() {
    local machine_id=$1
    local y_var=$2
    local strategy=$3
    local seq_len=$4

    local univariate_model_path="$EXP_DIR/out/$data_id/seq/univariate/${machine_id}_${y_var}/${strategy}/${machine_id}_${strategy}.pt"
    local multivariate_model_path="$EXP_DIR/out/$data_id/seq/multivariate/${machine_id}_${y_var}/${strategy}/${machine_id}_${strategy}.pt"

    if [[ ! -f $univariate_model_path ]]; then
        echo ">>> Univariate model not found: $univariate_model_path"
        return
    fi

    if [[ ! -f $multivariate_model_path ]]; then
        echo ">>> Multivariate model not found: $multivariate_model_path"
        return
    fi

    python plot_seq_uni_multi.py "$EXP_DIR/preprocessed_data/$machine_id/${machine_id}_75-300/${machine_id}_${y_var}.csv" \
        -y $y_var \
        -o "$EXP_DIR/out_plot/global/seq-uni-multi" \
        -s "$strategy" \
        -nl "$N_LABELS" \
        --univariate "$univariate_model_path" \
        --multivariate "$multivariate_model_path" \
        --seq_len "$seq_len" 
}


# run "m_25" "cpu" "gdumb" $SEQ_LEN

STRATEGIES=("naive" "ewc" "gss" "lwf" "agem" "gdumb")

for machine_id in "m_25" "m_881"; do
    for y_var in "cpu" "mem" "disk"; do
        for strategy in "${STRATEGIES[@]}"; do
            echo ">>> Plotting univariate vs multivariate for machine=$machine_id y=$y_var strategy=$strategy"
            run $machine_id $y_var $strategy $SEQ_LEN 
        done
    done
done