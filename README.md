## Install

- Dependencies

```bash
conda create -n cl python=3.10 pip
conda activate cl
conda install -c conda-forge mamba
pip install gorilla semver ruptures git+https://github.com/ContinualAI/avalanche.git@c2601fccec29bfa2f4ed692cb9955526111d56be
mamba install numpy=1.21 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install numpy=1.21 pandas black matplotlib scikit-learn scikit-multiflow torchmetrics seaborn -c conda-forge
pip install pydantic simplejson types-simplejson fastparquet

# for pipeline
mamba install -c conda-forge -c bioconda snakemake
  
# for development only
pip install black isort

# If protobuf error from wandb
pip uninstall wandb protobuf
pip install wandb protobuf
```

## Development

- Always check types!

```
mypy .
```

## Running
Note: If running google dataset, change num_classes in general.config to 2!
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
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/scenario/alibaba/container/classification/batch/split-chunks


PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/analysis/alibaba/container/

# PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/alibaba/container/classification/batch/split-chunks
```

## Scenario

### Split Chunks

Default chunk = 8

- No Retrain

```bash
# run model A
# rm -rf out/training/alibaba/container/classification/batch/split-chunks/no-retrain/A
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/container/classification/batch/split-chunks/no-retrain/A \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/a.yaml \
                  ./config/strategies/no_retrain/no_retrain.yaml

# run model B
# rm -rf out/training/alibaba/container/classification/batch/split-chunks/no-retrain/B
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/container/classification/batch/split-chunks/no-retrain/B \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/b.yaml \
                  ./config/strategies/no_retrain/no_retrain.yaml

PYTHONPATH=$PYTHONPATH:. snakemake \
    out/training/google/tes/classification/online/split-chunks/no-retrain \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/google/google.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/no_retrain/no_retrain.yaml
```

- Retrain from scratch each chunk

```bash
# rm -rf out/training/alibaba/container/classification/batch/split-chunks/from-scratch
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/container/classification/batch/split-chunks/from-scratch/A \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/a.yaml \
                  ./config/strategies/from_scratch/from_scratch.yaml

PYTHONPATH=$PYTHONPATH:. snakemake \
    out/training/google/mapped_nog/classification/online/split-chunks/from-scratch \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/google/google.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/from_scratch/from_scratch.yaml
```



- Retrain using GSS

```bash
# rm -rf out/training/alibaba/container/classification/batch/split-chunks/gss
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/container/classification/batch/split-chunks/gss \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/gss/gss.yaml

PYTHONPATH=$PYTHONPATH:. snakemake \
    out/training/google/mapped_nog/classification/online/split-chunks/gss \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/google/google.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/gss/gss_1k.yaml
```

- Retrain using GDumb

```bash
# rm -rf out/training/alibaba/container/classification/batch/split-chunks/gdumb
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/container/classification/batch/split-chunks/gdumb/A \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/a.yaml \
                  ./config/strategies/gdumb/gdumb.yaml

PYTHONPATH=$PYTHONPATH:. snakemake \
    out/training/google/mapped_nog/classification/online/split-chunks/gdumb \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/google/google.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/gdumb/gdumb.yaml
                  
```

- Retrain using EWC

```bash
# rm -rf out/training/alibaba/container/classification/batch/split-chunks/ewc
# PYTHONPATH=$PYTHONPATH:. snakemake \
#     --profile=swing out/training/alibaba/container/classification/batch/split-chunks/ewc \
#     --configfiles ./config/general.yaml \
#                   ./config/scenario/split_chunks.yaml \
#                   ./config/dataset/alibaba/alibaba.yaml \
#                   ./config/model/mlp.yaml \
#                   ./config/strategies/ewc/ewc.yaml

# rm -rf out/training/alibaba/container/classification/batch/split-chunks/ewc/B
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/container/classification/batch/split-chunks/ewc/B \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/b.yaml \
                  ./config/strategies/ewc/ewc.yaml

PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/google/mapped_nog/classification/online/split-chunks/ewc \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/google/google.yaml \
                  ./config/model/mlp.yaml \
                  ./config/strategies/ewc/ewc.yaml
```

``` graphing

PYTHONPATH=$PYTHONPATH:. snakemake out/eval/google/mapped_nog/classification/online/split-chunks -c1

```


## Drift Detection


```bash
# rm -rf out/training/alibaba/container/classification/batch/split-chunks/no-retrain
# PYTHONPATH=$PYTHONPATH:. snakemake \
#     --profile=swing out/training/alibaba/container/classification/batch/drift-detection/no-retrain \
#     --configfiles ./config/general.yaml \
#                   ./config/scenario/drift_detection.yaml \
#                   ./config/dataset/alibaba/alibaba.yaml \
#                   ./config/model/mlp.yaml \
#                   ./config/strategies/no_retrain/no_retrain.yaml
```

## Analyzing dataset

```
python -m src.preprocess.analyze_dataset
```
