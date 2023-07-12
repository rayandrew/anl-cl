## Install

- Dependencies

```bash
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
- `<FEATURE_ENGINEERING>` = `no-feature|feature-*` can be anything depends on the dataset, see `src.helpers.features`
- `<TRAINING>` = `batch|online`
- `<TASK>` = `classification`
- `<SCENARIO>` = `split-chunks|drift-detection`
- `<STRATEGY>` = `no-retrain|naive|from-scratch|gss|agem|gem|ewc|mas|si|lwf|gdumb`
- `<MODEL>` = `model-a|model-b`, see `src.helpers.model`
- `<..._CONFIG>` = You can put arbitrary config files in YAML, it will be merged if you specify the same keys

```bash
# running training
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/training/<DATASET>/<FILENAME>/<TRAINING>/<SCENARIO>/<MODEL>/<FEATURE_ENGINEERING>/<STRATEGY> \
    --configfiles <GENERAL_CONFIG> \
                  <DATASET_CONFIG> \
                  <SCENARIO_CONFIG> \
                  <MODEL_CONFIG> \
                  <STRATEGY_CONFIG> \
                  ...<ADDITIONAL_CONFIG>
```

#### Example

```bash
# run model A on `azure` dataset with filename "vmcpu.parquet" with no-retrain strategy and feature engineering A
# rm -rf out/training/azure/vmcpu/classification/batch/split-chunks/model-a/feats-a/no-retrain
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/azure/vmcpu/classification/batch/split-chunks/model-a/feature-a/no-retrain \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split_chunks.yaml \
                  ./config/dataset/azure/azure.yaml \
                  ./config/model/a.yaml \
                  ./config/strategies/no_retrain/no_retrain.yaml
```

### Evaluation

#### Scenario

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/scenario/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>
```

#### Model

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/model/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>/<MODEL>
```

#### Feature Engineering

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/feats/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>/<MODEL>/<FEATURE_ENGINEERING>
```

#### Strategy

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/strategy/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>/<MODEL>/<FEATURE_ENGINEERING>/<STRATEGY>
```
