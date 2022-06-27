#!/bin/bash

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3

N_CHANNELS="16 32 64 128"
# N_CHANNELS="16 32 64 64"
# N_CHANNELS="128 128 128 128"

((nvidia-smi &> /dev/null) && export DEVICE=${GPU_DEVICE:0}) || export DEVICE="cpu"

LEARNING_RATES=( 5 6 7 )
LATENT_DIMS=( 8 16 32 )

for LR in ${LEARNING_RATES[@]}; do
for LATENT_DIM in ${LATENT_DIMS[@]}; do
  python main.py \
    --n_channels_enc $N_CHANNELS \
    --n_channels_dec $N_CHANNELS \
    --latent_dim $LATENT_DIM \
    --w_kl 0 \
    --batch_size 256 \
    --learning_rate 1e-${LR} \
    --gpus ${GPU_DEVICE:-0} \
    $@
done
done
