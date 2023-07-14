# ANL/CL

## Environment 

- Dependencies

```bash
conda install mamba -n base -c conda-forge

mamba env create --file env-pinned.yaml -n acl
# or
# mamba env create --file env.yaml -n acl

conda activate acl

# If protobuf error from wandb
pip uninstall wandb protobuf
pip install wandb protobuf
```

- Update deps

```bash
conda activate acl
mamba env update --file=env.yaml
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
                  ./config/scenario/split-chunks/split-chunks.yaml \
                  ./config/dataset/azure/azure.yaml \
                  ./config/model/a.yaml \
                  ./config/strategies/no-retrain/no-retrain.yaml
```

### Evaluation

#### Compare Scenario + Model + Strategy + Feature Engineering

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/dataset/<DATASET>/<FILEPATH>/<TRAINING>
```

#### Compare Model + Strategy + Feature Engineering (same scenario)

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/scenario/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>
```

#### Compare Strategy + Feature Engineering (same scenario and model)

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/model/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>/<MODEL>
```

#### Compare Strategy (same scenario, model and feature engineering) 

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/feature/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>/<MODEL>/<FEATURE_ENGINEERING>
```

#### Single Plot

```bash
PYTHONPATH=$PYTHONPATH:. snakemake --profile=swing out/evaluation/single/<DATASET>/<FILEPATH>/<TRAINING>/<SCENARIO>/<MODEL>/<FEATURE_ENGINEERING>/<STRATEGY>
```

## Development

- Always check types!

```
mypy .
```

- Install `pre-commit`

```
pre-commit install
```

## Azure VMCPU

### Run Training

#### Split Chunks

- No Retraining

```bash
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/azure/vmcpu/classification/batch/split-chunks/model-a/feature-a/no-retrain \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split-chunks/split-chunks.yaml \
                  ./config/dataset/azure/azure.yaml \
                  ./config/model/a.yaml \
                  ./config/strategies/no-retrain/no-retrain.yaml
```

- GSS

```bash
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/azure/vmcpu/classification/batch/split-chunks/model-a/feature-a/gss \
    --configfiles ./config/general.yaml \
                  ./config/scenario/split-chunks/split-chunks.yaml \
                  ./config/dataset/azure/azure.yaml \
                  ./config/model/a.yaml \
                  ./config/strategies/gss/gss.yaml
```

#### Drift Detection

- EWC + Voting

```bash
PYTHONPATH=$PYTHONPATH:. snakemake \
    --profile=swing out/training/azure/vmcpu/classification/batch/drift-detection/model-a/feature-a/ewc \
    --configfiles ./config/general.yaml \
                  ./config/scenario/drift-detection.yaml \
                  ./config/dataset/azure/azure.yaml \
                  ./config/model/a.yaml \
                  ./config/strategies/ewc/ewc.yaml \
                  ./config/drift-detection/voting/azure-vmcpu.yaml
```

### Evaluation

```bash
PYTHONPATH=$PYTHONPATH:. snakemake -c4 \
    out/evaluation/feature/azure/vmcpu/classification/batch/split-chunks/model-a/feature-a \
    out/evaluation/feature/azure/vmcpu/classification/batch/drift-detection/model-a/feature-a \
    out/evaluation/dataset/azure/vmcpu/classification/batch
```