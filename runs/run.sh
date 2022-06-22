#!/bin/bash

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3

# N_CHANNELS="16 32 64 128"
# N_CHANNELS="16 32 64 64"
N_CHANNELS="128 128 128 128"

((nvidia-smi &> /dev/null) && export DEVICE=${GPU_DEVICE:0}) || export DEVICE="cpu"

python main.py \
  -c config_files/config_folded_c_and_s.yaml \
  --n_channels_enc $N_CHANNELS \
  --n_channels_dec $N_CHANNELS \
  --latent_dim 32 \
  --w_kl 0 \
  --batch_size 512 \
  --learning_rate 0.00001 \
  --cardiac_dataset.meshes_file "data/cardio/procrustes_transforms_35k.pkl" \
  --cardiac_dataset.procrustes_transforms_file "data/cardio/meshes_files.npy" \
  --gpus ${GPU_DEVICE:-0} \
  $@

