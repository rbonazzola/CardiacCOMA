#!/bin/bash

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 2 1024x768x24 > /dev/null 2>&1 &
((nvidia-smi &> /dev/null) && export DEVICE=${GPU_DEVICE:0}) || export DEVICE="cpu"

MLFLOW_EXPERIMENT="Cardiac - ED"
N_CHANNELS=${N_CHANNELS:-"16 32 64 128"}
PARTITION_LENGTHS=${PARTITION_LENGTHS:-"5120 1024 1024 -1"}
WKLS=${WKLS:-"1e-3"}
BATCHSIZE=${BATCHSIZE:-64}

if [ -z "${LEARNING_RATES}" ]; then
  echo "lr not set"
  LEARNING_RATES=( "1e-5" )
fi 

if [ -z "${LATENT_DIMS}" ]; then
  LATENT_DIMS=( 8 )
fi 

N_REP=5

for LR in ${LEARNING_RATES[@]}; do
for LATENT_DIM in ${LATENT_DIMS[@]}; do
for WKL in ${WKLS[@]}; do
for i in `seq 1 $N_REP`; do
  python main.py \
    --n_channels_enc $N_CHANNELS \
    --n_channels_dec $N_CHANNELS \
    --latent_dim $LATENT_DIM \
    --w_kl $WKL \
    --batch_size $BATCHSIZE \
    --learning_rate ${LR} \
    --gpus ${GPU_DEVICE:-0} \
    --partition_lengths ${PARTITION_LENGTHS} \
    --seed $i \
    --min_epochs 20 \
    --max_epochs 10000 \
    $@
done
done
done
done
