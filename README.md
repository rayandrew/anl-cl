## Install

- Dependencies

```bash
conda create -n cl python=3.10 pip
conda activate cl
mamba install -c conda-forge mamba
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# pip install git+https://github.com/ContinualAI/avalanche.git@0088c3092af918ac2c16d3f945be8dd62415a01c
pip install git+https://github.com/ContinualAI/avalanche.git@c2601fccec29bfa2f4ed692cb9955526111d56be
mamba install pandas black matplotlib scikit-learn scikit-multiflow numpy seaborn -c conda-forge
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
snakemake -cN --configfile ./config/offline_alibaba.yaml ./config/dd/voting.yaml
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
