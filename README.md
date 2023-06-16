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

# running scenario
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing <OUTPUT> \
    --configfiles <GENERAL_CONFIG> \
                  <DATASET_CONFIG> \
                  <SCENARIO_CONFIG> \
                  <MODEL_CONFIG> \
                  <STRATEGY_CONFIG>

# running evaluation
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/eval/alibaba/chunk-0/classification/batch/split-chunks
```

## Scenario

### Split Chunks

Default chunk = 8

- No Retrain

```bash
# rm -rf out/training/alibaba/chunk-0/classification/batch/split-chunks/no-retrain
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/chunk-0/classification/batch/split-chunks/no-retrain \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/no_retrain/no_retrain.yaml
```

- Retrain from scratch each chunk

```bash
# rm -rf out/training/alibaba/chunk-0/classification/batch/split-chunks/from-scratch
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/chunk-0/classification/batch/split-chunks/from-scratch \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/from_scratch/from_scratch.yaml
```

- Retrain using GSS

```bash
# rm -rf out/training/alibaba/chunk-0/classification/batch/split-chunks/gss
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/chunk-0/classification/batch/split-chunks/gss \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/gss/gss.yaml
```

- Retrain using GDumb

```bash
# rm -rf out/training/alibaba/chunk-0/classification/batch/split-chunks/gdumb
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/chunk-0/classification/batch/split-chunks/gdumb \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/gdumb/gdumb.yaml
```

- Retrain using EWC

```bash
# rm -rf out/training/alibaba/chunk-0/classification/batch/split-chunks/ewc
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/chunk-0/classification/batch/split-chunks/ewc \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/ewc/ewc.yaml
```

## Analyzing dataset

```
python -m src.preprocess.analyze_dataset
```
