#!/bin/bash
#SBATCH --time=01-00:30
#SBATCH --gpus=h100
#SBATCH --cpus-per-task=12
#SBATCH --mem=32gb

export WANDB_API_KEY="efe870a3e39f7ac1dde74d2cfedd1fbb0287cbce"
. setup.sh
flwr run ./
