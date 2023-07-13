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
# inside conda `base` env
conda install mamba -n base -c conda-forge

mamba env create --file env-pinned.yaml
# or
# mamba env create --file env.yaml

conda activate acl

# If protobuf error from wandb
pip uninstall wandb protobuf
pip install wandb protobuf

# update deps
# mamba env update --file=env.yaml
```

## Development

- Always check types!

```
mypy .
```

## Running

### Training

#### Glossary

**SUBJECT TO CHANGE**

See `workflow/rules/pipeline.smk`

- `<DATASET>` = `alibaba|google|cori|azure`
- `<FILENAME>` = dataset filename that should exist in `raw_data/<DATASET>/<FILEPATH>.parquet`
- `<FEATS>` = `feats-*` can be anything depends on the dataset, see `src.helpers.feats`
- `<TRAINING>` = `batch|online`
- `<TASK>` = `classification`
- `<SCENARIO>` = `split-chunks|drift-detection`
- `<STRATEGY>` = `no-retrain|naive|from-scratch|gss|agem|gem|ewc|mas|si|lwf|gdumb`
- `<MODEL>` = `model-a|model-b`, see `src.helpers.model`
- `<..._CONFIG>` = You can put arbitrary config files in YAML, it will be merged if you specify the same keys

```bash
# running training
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/training/<DATASET>/<FILENAME>/<TRAINING>/<SCENARIO>/<MODEL>/<STRATEGY>/<FEATS> \
    --configfiles <GENERAL_CONFIG> \
                  <DATASET_CONFIG> \
                  <SCENARIO_CONFIG> \
                  <MODEL_CONFIG> \
                  <STRATEGY_CONFIG> \
                  ...<ADDITIONAL_CONFIG>
```

#### Example

```bash
# run model A on alibaba dataset with filename "container".parquet with no-retrain strategy and feature engineering A
# rm -rf out/training/alibaba/container/classification/batch/split-chunks/no-retrain/A
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/alibaba/container/classification/batch/split-chunks/model-a/no-retrain/feats-a \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/alibaba/alibaba.yaml \
                  ./config/model/a.yaml \
                  ./config/strategies/no_retrain/no_retrain.yaml
```

### Evaluation

#### Scenario

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
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/scenario/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>
```

#### Model

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/scenario/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>/<MODEL>
```

#### Strategy

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/scenario/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>/<MODEL>/<STRATEGY>
```

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
