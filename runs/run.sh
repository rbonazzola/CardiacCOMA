#!/bin/bash

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 2 1024x768x24 > /dev/null 2>&1 &
((nvidia-smi &> /dev/null) && export DEVICE=${GPU_DEVICE:0}) || export DEVICE="cpu"

N_CHANNELS=${N_CHANNELS:-"16 32 64 128"}
PARTITION_LENGTHS=${PARTITION_LENGTHS:-"5120 1024 1024 -1"}
MLFLOW_EXPERIMENT="Cardiac - ED"
WKL=${WKL:-1e-3}
BATCHSIZE=${BATCHSIZE:-128}

if [ -z "${LEARNING_RATES}" ]; then
  echo "lr not set"
  LEARNING_RATES=( 5 )
fi 

if [ -z ${LATENT_DIMS} ]; then
  LATENT_DIMS=( 8 )
fi 

N_REP=1

#MLFLOW_EXPERIMENT="test"

for LR in ${LEARNING_RATES[@]}; do
for LATENT_DIM in ${LATENT_DIMS[@]}; do
for i in `seq 1 $N_REP`; do
  python main.py \
    --n_channels_enc $N_CHANNELS \
    --n_channels_dec $N_CHANNELS \
    --latent_dim $LATENT_DIM \
    --w_kl $WKL \
    --batch_size $BATCHSIZE \
    --learning_rate 1e-${LR} \
    --gpus ${GPU_DEVICE:-0} \
    --partition_lengths ${PARTITION_LENGTHS} \
    --seed $i \
    --min_epochs 20 \
    $@
done
done
done
