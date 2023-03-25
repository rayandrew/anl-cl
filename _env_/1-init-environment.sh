#!/bin/bash

module load cuda/11.4.0-gqbcqie 
module load openmpi/4.1.4-cuda-ucx
# module load anaconda3/2023-01-11

# Activate conda env
source /gpfs/fs1/soft/swing/manual/anaconda3/2023-01-11/mconda3/etc/profile.d/conda.sh
conda activate /home/ac.rayandrew/.conda/envs/cl

# python -c "import torch; print(torch.cuda.is_available())"