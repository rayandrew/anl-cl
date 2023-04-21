# Argonne CL

## Install

- CUDA

```bash
# run this in Chameleon node
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get remove --purge '^libnvidia-.*'
sudo apt-get remove --purge '^cuda-.*'
sudo apt-get install linux-headers-$(uname -r)
```

Next, we need to download the latest version of CUDA that is supported by PyTorch (at the time of this project: [11.7][https://developer.nvidia.com/cuda-11-7-0-download-archive])
- Dependencies

```bash
conda create -n cl python=3.10 pip
conda activate cl
mamba install -c conda-forge mamba
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install git+https://github.com/ContinualAI/avalanche.git@0088c3092af918ac2c16d3f945be8dd62415a01c
mamba install pandas black matplotlib scikit-learn scikit-multiflow numpy seaborn -c conda-forge
pip install texttable
pip install sacred # for configuration management

# for development only
pip install black isort

mamba install -c conda-forge hydra-core
mamba install -c conda-forge dvc
# mamba install -c conda-forge neptune neptune-sacred
pip install -U "neptune[sacred]"
```

## Running

```bash
python main.py -f ./data/m_881.csv -m m_881_gdumb -x 3 -s gdumb -o ./out/m_881_mem/gdumb/ -y mem_util_percent
python eval.py -f ./data/m_881.csv -o out/m_881_mem/gdumb -m ./out/m_881_mem/gdumb/m_881_gdumb.pt -y mem_util_percent --plot

python main.py -f ./data/m_881.csv -m m_881_gss -x 3 -s gss -o ./out/m_881_mem/gss/ -y mem_util_percent
python eval.py -f ./data/m_881.csv -o out/m_881_mem/gss -m ./out/m_881_mem/gss/m_881_gss.pt -y mem_util_percent --plot
```
