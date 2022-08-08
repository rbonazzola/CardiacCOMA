#!/bin/bash

LOG10LR=4
LATENT_DIM=16

# N_CHANNELS="16 32 64 64"
N_CHANNELS="128 128 128 128"
#N_CHANNELS="1024 512 256 128"

#PARTITION_LENGTHS="5120 1024 1024 -1"
PARTITION_LENGTHS="1024 1024 1024 4096"

MLFLOW_EXPERIMENT="test"
# MLFLOW_EXPERIMENT="Cardiac - ED"

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
((nvidia-smi &> /dev/null) && export DEVICE=${GPU_DEVICE:0}) || export DEVICE="cpu"

python main.py \
  --n_channels_enc $N_CHANNELS \
  --n_channels_dec $N_CHANNELS \
  --latent_dim $LATENT_DIM \
  --w_kl 1e-1 \
  --batch_size 32 \
  --learning_rate 2e-${LOG10LR} \
  --gpus ${GPU_DEVICE:-1} \
  --partition_lengths ${PARTITION_LENGTHS} \
  --max_epochs 2 \
  --precision 16 \
  --mlflow_experiment "${MLFLOW_EXPERIMENT}" \
  $@
