## Install

- Dependencies

```bash
conda create -n cl python=3.9 pip
conda activate cl
# Voting library need low numpy ver
pip install numpy==1.19 
conda install -c conda-forge mamba
pip install gorilla hydra-core
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install git+https://github.com/ContinualAI/avalanche.git@c2601fccec29bfa2f4ed692cb9955526111d56be
mamba install pandas black matplotlib scikit-learn scikit-multiflow seaborn -c conda-forge
mamba install -c conda-forge torchmetrics
pip install texttable
pip install semver

# for pipeline
mamba install -c conda-forge -c bioconda snakemake
  
# for development only
pip install black isort
pip install wandb
```

## Running

```bash
export PYTHONPATH=$PYTHONPATH:.
snakemake -c1 --configfile ./config/dataset/alibaba.yaml ./config/dd/voting.yaml
snakemake -c1 --configfile ./config/dataset/google.yaml ./config/dd/voting.yaml
# change N to number of concurrency that you want
```

- Slurm

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing --jobs 2 --directory `realpath -s .`
```

## Analyzing dataset

```
python -m src.preprocess.analyze_dataset
```
