## Install

- Dependencies

```bash
conda create -n cl python=3.10 pip
conda activate cl
conda install -c conda-forge mamba
pip install gorilla semver ruptures git+https://github.com/ContinualAI/avalanche.git@c2601fccec29bfa2f4ed692cb9955526111d56be
mamba install numpy=1.21 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install numpy=1.21 pandas black matplotlib scikit-learn scikit-multiflow torchmetrics seaborn -c conda-forge
pip install pydantic

# for pipeline
mamba install -c conda-forge -c bioconda snakemake
  
# for development only
pip install black isort

# If protobuf error from wandb
pip uninstall wandb protobuf
pip install wandb protobuf
```

## Running

```bash
# Important for module not found!
export PYTHONPATH=$PYTHONPATH:.
snakemake -cN
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
