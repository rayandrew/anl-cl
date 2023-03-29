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
# SEQ_TYPE="multivariate"
SEQ_TYPE="univariate"

data_id="global"
if [[ $LOCAL == 1 ]]; then
    data_id="local"
fi

# python plot_seq_non_seq.py ./preprocessed_data/m_25/m_25_75-300/m_25_disk.csv -y disk -o ./out_plot/seq-non-seq -s gss -nl 10 --seq ./out/global/seq/m_25_disk/gss/m_25_gss.pt --non_seq ./out/global/non-seq/m_25_disk/gss/m_25_gss.pt 
run() {
    local machine_id=$1
    local y_var=$2
    local strategy=$3
    local seq_len=$4
    local seq_type=$5

    local seq_model_path="$EXP_DIR/out/$data_id/seq/${seq_type}/${machine_id}_${y_var}/${strategy}/${machine_id}_${strategy}.pt"
    local non_seq_model_path="$EXP_DIR/out/$data_id/non-seq/${machine_id}_${y_var}/${strategy}/${machine_id}_${strategy}.pt"

    if [[ ! -f $seq_model_path ]]; then
        echo ">>> Sequential model not found: $seq_model_path"
        return
    fi

    if [[ ! -f $non_seq_model_path ]]; then
        echo ">>> Non-sequential model not found: $non_seq_model_path"
        return
    fi

    if [[ $seq_type == "univariate" ]]; then
        python plot_seq_non_seq.py "$EXP_DIR/preprocessed_data/$machine_id/${machine_id}_75-300/${machine_id}_${y_var}.csv" \
            -y $y_var \
            -o "$EXP_DIR/out_plot/global/${seq_type}-seq-non-seq" \
            -s "$strategy" \
            -nl "$N_LABELS" \
            --seq "$seq_model_path" \
            --non_seq "$non_seq_model_path" \
            --seq_len "$seq_len" \
            --univariate
            return
    fi

    python plot_seq_non_seq.py "$EXP_DIR/preprocessed_data/$machine_id/${machine_id}_75-300/${machine_id}_${y_var}.csv" \
        -y $y_var \
        -o "$EXP_DIR/out_plot/global/${seq_type}-seq-non-seq" \
        -s "$strategy" \
        -nl "$N_LABELS" \
        --seq "$seq_model_path" \
        --non_seq "$non_seq_model_path" \
        --seq_len "$seq_len" 
}


# run "m_25" "cpu" "gdumb" $SEQ_LEN $SEQ_TYPE

STRATEGIES=("naive" "ewc" "gss" "lwf" "agem" "gdumb")

for machine_id in "m_25" "m_881"; do
    for y_var in "cpu" "mem" "disk"; do
        for strategy in "${STRATEGIES[@]}"; do
            echo ">>> Plotting seq vs non-seq for machine=$machine_id y=$y_var strategy=$strategy"
            run $machine_id $y_var $strategy $SEQ_LEN $SEQ_TYPE
        done
    done
done