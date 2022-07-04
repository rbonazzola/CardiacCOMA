#!/bin/bash

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

N_CHANNELS="16 32 64 128"
# N_CHANNELS="16 32 64 64"
# N_CHANNELS="128 128 128 128"

((nvidia-smi &> /dev/null) && export DEVICE=${GPU_DEVICE:0}) || export DEVICE="cpu"

LOG10LR=5
LATENT_DIM=8

python main.py \
  --n_channels_enc $N_CHANNELS \
  --n_channels_dec $N_CHANNELS \
  --latent_dim $LATENT_DIM \
  --w_kl 0 \
  --batch_size 256 \
  --learning_rate 1e-${LOG10LR} \
  --gpus ${GPU_DEVICE:-0} \
  --partition_lengths 256 256 256 512 \
  --max_epochs 2 \
  --mlflow_experiment test \
  $@
