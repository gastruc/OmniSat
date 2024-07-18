# OmniSat: Self-Supervised Modality Fusion for Earth Observation (ECCV 2024)

[![python](https://img.shields.io/badge/-Python_3.8+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.2+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

[//]: # ([![Paper]&#40;https://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)
[//]: # ([![Conference]&#40;https://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/paper/2020&#41;)


Official implementation for <br> [_OmniSat: Self-Supervised Modality Fusion for Earth Observation_](https://arxiv.org/pdf/2404.08351.pdf) <br>

## Description

### Abstract

We introduce OmniSat, a novel architecture that exploits the spatial alignment between multiple EO modalities to learn expressive multimodal representations without labels. We demonstrate the advantages of combining modalities of different natures across three downstream tasks (forestry, land cover classification, and crop mapping), and  propose two augmented datasets with new modalities: PASTIS-HD and TreeSatAI-TS.

<p align="center">
  <img src="https://github.com/gastruc/OmniSat/assets/1902679/9fc20951-1cac-4891-b67f-53ed5e0675ad" width="500" height="250">
</p>

### Datasets

  
| Dataset name  |             Modalities                   |      Labels         |     Link      
| ------------- | ---------------------------------------- | ------------------- | ------------------- |
| PASTIS-HD     | **SPOT 6-7 (1m)** + S1/S2 (30-140 / year)| Crop mapping (0.2m) |    [huggingface](https://huggingface.co/datasets/IGNF/PASTIS-HD) or [zenodo](https://zenodo.org/records/10908628) |
| TreeSatAI-TS  | Aerial (0.2m) + **S1/S2 (10-70 / year)** |   Forestry (60m)    |   [huggingface](https://huggingface.co/datasets/IGNF/TreeSatAI-Time-Series) |
| FLAIR         |   aerial (0.2m) + S2 (20-114 / year)     |  Land cover (0.2m)  |  [huggingface](https://huggingface.co/datasets/IGNF/FLAIR) |


<p align="center">
  <img src="https://github.com/user-attachments/assets/18acbb19-6c90-4c9a-be05-0af24ded2052" width="500" height="250">
</p>

### Results

We perform experiments with 100% and 10-20% of labels. See below, the F1 Score results on 100% of the training data with all modalities available:

|   F1 Score All Modalities   | UT&T | Scale-MAE |   DOFA   | OmniSat (no pretraining) | OmniSat (with pretraining) |
| ------------- | ---- | --------- | -------- |------------------------- | -------------------------- |
| PASTIS-HD     | 53.5 |   42.2    |   55.7   |          59.1            |          **69.9**          |
| TreeSatAI-TS  | 56.7 |   60.4    |   71.3   |          73.3            |          **74.2**          |
| FLAIR         | 48.8 |   70.0    | **74.9** |          70.0            |            73.4            |

OmniSat also improves performance even when only one modality is available for inference.
F1 Score results on 100% of the training data with only S2 data available:

|   F1 Score S2 only   | UT&T | Scale-MAE | DOFA | OmniSat (no pretraining) | OmniSat (with pretraining) |
| ------------- | ---- | --------- | -----|------------------------- | -------------------------- |
| PASTIS-HD     | 61.3 |   46.1    | 53.4 |           60.1           |          **70.8**          |
| TreeSatAI-TS  | 57.0 |   31.5    | 39.4 |           49.7           |          **62.9**          |
| FLAIR         | 62.0 |   61.0    | 61.0 |         **65.4**         |          **65.4**          |



### Efficiency

We report the best performance of different models between TreeSatAI and TreeSatAI-TS, with pre-training and fine-tuning using 100% of labels. The area of the markers is proportional to the training time, broken down in pre-training and fine-tuning when applicable

<p align="center">
   <img src="https://github.com/user-attachments/assets/0e6a378a-024a-4224-ad1d-fa7171df5adf" width="550" height="250">
</p>

## Project Structure

The directory structure of new project looks like this:

```
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── dataset                  <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── exp                      <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── config.yaml            <- Main config for training
│   └── eval.yaml              <- Main config for evaluation
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── src                    <- Source code
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   ├── train_pastis_20.py       <- Run training on 20% pastis dataset
│   └── train.py                 <- Run training
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

## Getting the data

## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/ashleve/lightning-hydra-template
cd lightning-hydra-template

# [OPTIONAL] create conda environment
conda create -n omni python=3.9
conda activate omni

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# Create data folder where you can put your datasets
mkdir data
# Create logs folder
mkdir logs
```

## Usage

Every experience of the paper has its own config. Feel free to explore configs/exp folder

```bash
python src/train.py exp=TSAITS_OmniSAT #to run OmniSAT pretraining on TreeSatAI-TS
#trainer.devices=X to change the number of GPU you want to train on
#trainer.num_workers=16 to change the num_workers available
#dataset.global_batch_size=16 to change global batch size (ie batch size that will be distributed across all GPUS)
#offline=True to run in offline mode from wandb
#max_epochs=1 to change the number maximum of epochs

python src/train.py exp=TSAITS_OmniSAT #to run OmniSAT finetuning on TreeSatAI-TS
#model.name=OmniSAT_MM  to change model name for logging
#partition=1.0 to change the percentage on training data you want to use

# All these parameters and more can be changed from the config file
```

To run 20% experiments on PASTIS-HD, you have to run

```bash
python src/train_pastis_20.py exp=Pastis_ResNet #to run a ResNet on PASTIS-HD
#partition parameter does not change anything on PASTIS-HD
```
## Citation

To refer to this work, please cite
```
@article{astruc2024omnisat,
  title={Omni{S}at: {S}elf-Supervised Modality Fusion for {E}arth Observation},
  author={Astruc, Guillaume and Gonthier, Nicolas and Mallet, Clement and Landrieu, Loic},
  journal={ECCV},
  year={2024}
}
```


## Acknowledgements
- This project was built using [Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template).
- The original code of [TreeSat](https://git.tu-berlin.de/rsim/treesat_benchmark)
- The transformer structures come from [timm](https://github.com/huggingface/pytorch-image-models)
- The implementation of UT&T come from the [FLAIR 2 challenge repository](https://github.com/IGNF/FLAIR-2)
- The implementation of SatMAE and ScaleMAE come from [USat](https://github.com/stanfordmlgroup/USat)
- The Croma implementation comes from [Croma](https://github.com/antofuller/CROMA)
- The Relative Positional Encoding implementation come from [iRPE](https://github.com/microsoft/Cream/tree/main/iRPE)
- The Pse, ltae, UTAE come from [utae-paps](https://github.com/VSainteuf/utae-paps/tree/main)

<br>
