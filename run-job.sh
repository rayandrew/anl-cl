#!/bin/bash
#SBATCH --job-name=acl    # create a short name for your job
#SBATCH --account=FASTBAYES
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=./stdout/stdout_run_acl.%j
#SBATCH --output=./stdout/stdout_run_acl.%j

# User Configuration
EXP_DIR=$PWD
INIT_SCRIPT=$PWD/_env_/1-init-environment.sh
SLURM_CPUS_PER_TASK=5
SLURM_GPUS_PER_TASK=1

# Initialize environment
source $INIT_SCRIPT

# Run experiment here
# MACHINE_ID="m_881"
# N_EXP=3
MACHINE_ID="m_25"
N_EXP=11
Y_VAR="disk"
STRATEGY="gss"
OUT_FOLDER="out_mu_seq"
SEQ_LEN=5
EVAL_ONLY=0

if [[ $OUT_FOLDER == "out_mu" ]]; then
    if [[ $EVAL_ONLY == 0 ]]; then
        python main.py -f "$EXP_DIR/data/mu_dist/$MACHINE_ID.csv" \
                       -m "${MACHINE_ID}_${STRATEGY}" \
                       -x $N_EXP \
                       -s $STRATEGY \
                       -o "$EXP_DIR/$OUT_FOLDER/${MACHINE_ID}_${Y_VAR}/${STRATEGY}/" \
                       -y $Y_VAR
    fi
    python eval.py -f "$EXP_DIR/data/mu_dist/$MACHINE_ID.csv" \
                   -o "$EXP_DIR/$OUT_FOLDER/${MACHINE_ID}_${Y_VAR}/${STRATEGY}" \
                   -m "$EXP_DIR/$OUT_FOLDER/${MACHINE_ID}_${Y_VAR}/${STRATEGY}/${MACHINE_ID}_${STRATEGY}.pt" \
                   -y $Y_VAR \
                   --plot
elif [[ $OUT_FOLDER == "out_mu_seq" ]]; then
    if [[ $EVAL_ONLY == 0 ]]; then
        python main.py -f "$EXP_DIR/out_preprocess/$MACHINE_ID/$MACHINE_ID.csv" \
                       -m "${MACHINE_ID}_${STRATEGY}" \
                       -x $N_EXP \
                       -s $STRATEGY \
                       -o "$EXP_DIR/$OUT_FOLDER/${MACHINE_ID}_${Y_VAR}/${STRATEGY}/" \
                       -y $Y_VAR \
                       --seq --seq_len $SEQ_LEN
    fi
    python eval.py -f "$EXP_DIR/out_preprocess/$MACHINE_ID/$MACHINE_ID.csv" \
                   -o "$EXP_DIR/$OUT_FOLDER/${MACHINE_ID}_${Y_VAR}/${STRATEGY}" \
                   -m "$EXP_DIR/$OUT_FOLDER/${MACHINE_ID}_${Y_VAR}/${STRATEGY}/${MACHINE_ID}_${STRATEGY}.pt" \
                   -y $Y_VAR \
                   --plot \
                   --seq --seq_len $SEQ_LEN
elif [[ $OUT_FOLDER == "out" ]]; then
    if [[ $EVAL_ONLY == 0 ]]; then
        python main.py -f "$EXP_DIR/data/$MACHINE_ID.csv" \
                       -m "${MACHINE_ID}_${STRATEGY}" \
                       -x $N_EXP \
                       -s $STRATEGY \
                       -o "$EXP_DIR/$OUT_FOLDER/${MACHINE_ID}_${Y_VAR}/${STRATEGY}/" \
                       -y $Y_VAR
    fi
    python eval.py -f "$EXP_DIR/data/$MACHINE_ID.csv" \
                   -o "$EXP_DIR/$OUT_FOLDER/${MACHINE_ID}_${Y_VAR}/${STRATEGY}" \
                   -m "$EXP_DIR/$OUT_FOLDER/${MACHINE_ID}_${Y_VAR}/${STRATEGY}/${MACHINE_ID}_${STRATEGY}.pt" \
                   -y $Y_VAR \
                   --plot   
fi