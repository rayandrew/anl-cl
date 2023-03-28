#!/bin/bash
#SBATCH --job-name=acl    
#SBATCH --account=FASTBAYES
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00          
#SBATCH --output=./stdout/stdout_run_acl.%j
#SBATCH --output=./stdout/stdout_run_acl.%j

set -e

# User Configuration
EXP_DIR=$PWD
INIT_SCRIPT=$PWD/scripts/1-init-environment.sh
ALIBABA_MU="/lcrc/project/FastBayes/rayandrew/machine_usage"

# Initialize environment
source $INIT_SCRIPT

python "$EXP_DIR/dataset.py" \
    -d "$ALIBABA_MU/m_25.csv" \
    -y disk \
    -m predict \
    --univariate --seq